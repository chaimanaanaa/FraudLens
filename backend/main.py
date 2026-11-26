"""
FraudLens — FastAPI backend
Exposes 4 agent routes called sequentially by n8n:
  POST /extract   — Vision Agent (Gemini Vision)
  POST /retrieve  — RAG Agent (Gemini Embeddings + Qdrant)
  POST /analyze   — Analysis Agent (Gemini)
  POST /validate  — Validator Agent (Gemini + guardrails + 1-retry)
"""
import logging
import os
from pathlib import Path

import yaml
from google import genai
from google.genai import types
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.models import (
    ExtractRequest, ExtractResponse,
    RetrieveRequest, RetrieveResponse,
    AnalyzeRequest, AnalyzeResponse,
    ValidateRequest, ValidateResponse,
)
from backend.agents.extract import run_extract
from backend.agents.retrieve import run_retrieve
from backend.agents.analyze import run_analyze
from backend.agents.validate import run_validate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fraudlens")


def _load_settings() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["gemini"]["api_key"] = os.getenv("GEMINI_API_KEY", cfg["gemini"].get("api_key", ""))
    cfg["gemini"]["vision_model"] = os.getenv("GEMINI_VISION_MODEL", cfg["gemini"]["vision_model"])
    cfg["gemini"]["llm_model"] = os.getenv("GEMINI_LLM_MODEL", cfg["gemini"]["llm_model"])
    cfg["gemini"]["embed_model"] = os.getenv("GEMINI_EMBED_MODEL", cfg["gemini"]["embed_model"])
    cfg["qdrant"]["host"] = os.getenv("QDRANT_HOST", cfg["qdrant"]["host"])
    cfg["qdrant"]["port"] = int(os.getenv("QDRANT_PORT", cfg["qdrant"]["port"]))
    cfg["qdrant"]["collection"] = os.getenv("QDRANT_COLLECTION", cfg["qdrant"]["collection"])
    return cfg


settings = _load_settings()

# Single Gemini client shared by all agents. SDK-level retries are disabled:
# backend/agents/gemini.py retries and falls back to other models itself.
settings["gemini"]["_client"] = genai.Client(
    api_key=settings["gemini"]["api_key"],
    http_options=types.HttpOptions(
        timeout=int(settings["gemini"].get("timeout_seconds", 60) * 1000),
        retry_options=types.HttpRetryOptions(attempts=1),
    ),
)

prompt_dir = Path(__file__).parent.parent / "prompts"

app = FastAPI(
    title="FraudLens API",
    description="4-agent fraud detection pipeline: extract → retrieve → analyze → validate",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "fraudlens-backend"}


def _fail(route: str, exc: Exception) -> HTTPException:
    status = 400 if isinstance(exc, ValueError) else 500
    log.exception("%s failed", route)
    return HTTPException(status_code=status, detail=f"{route}: {type(exc).__name__}: {exc}")


@app.post("/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest) -> ExtractResponse:
    try:
        return await run_extract(request, settings, prompt_dir)
    except Exception as exc:
        raise _fail("extract", exc)


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    try:
        return await run_retrieve(request, settings)
    except Exception as exc:
        raise _fail("retrieve", exc)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        return await run_analyze(request, settings, prompt_dir)
    except Exception as exc:
        raise _fail("analyze", exc)


@app.post("/validate", response_model=ValidateResponse)
async def validate(request: ValidateRequest) -> ValidateResponse:
    try:
        return await run_validate(request, settings, prompt_dir)
    except Exception as exc:
        raise _fail("validate", exc)
