"""
Analysis Agent — calls Gemini to classify the message.
Uses prompt v2/analyze.txt (chain-of-thought + few-shot + structured JSON output).
"""
import json
import re
from pathlib import Path
from typing import Any, List

from google import genai
from google.genai import types

from backend.agents.gemini import ContentBlocked, generate, model_chain, thinking
from backend.models import AnalyzeRequest, AnalyzeResponse, Passage

REFUND_KEYWORDS = ["remboursement", "rembourser", "récupérer votre argent", "refund", "reimbursement"]


def _load_prompt(prompt_dir: Path, version: str) -> str:
    return (prompt_dir / version / "analyze.txt").read_text(encoding="utf-8").strip()


def _format_context(passages: List[Passage]) -> str:
    if not passages:
        return "No similar fraud patterns found in the knowledge base."
    lines = []
    for i, p in enumerate(passages, 1):
        label_tag = f"[{p.label.upper()}]" if p.label != "unknown" else ""
        lines.append(f"{i}. {label_tag} (score={p.score:.2f}) {p.text}")
    return "\n".join(lines)


def _extract_json(raw: str) -> dict:
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from Gemini output: {raw[:300]}")


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(v) for v in value if str(v).strip()]


def _apply_guardrails(data: dict) -> dict:
    try:
        confidence = float(data.get("confidence_score", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    data["confidence_score"] = max(0.0, min(confidence, 0.95))

    verdict = str(data.get("verdict", "SUSPICIOUS")).upper()
    data["verdict"] = verdict if verdict in {"SCAM", "SUSPICIOUS", "LEGITIMATE"} else "SUSPICIOUS"

    fraud_type = data.get("fraud_type")
    data["fraud_type"] = None if fraud_type in (None, "", "null", "none") else str(fraud_type)

    data["reasons"] = _as_str_list(data.get("reasons")) or ["Analysis inconclusive"]
    data["cited_sources"] = _as_str_list(data.get("cited_sources"))
    actions = [
        a for a in _as_str_list(data.get("recommended_actions"))
        if not any(kw in a.lower() for kw in REFUND_KEYWORDS)
    ]
    data["recommended_actions"] = actions or ["Exercise caution"]
    return data


def _blocked_verdict(passages: List[Passage]) -> dict:
    """Conservative verdict when the provider refuses to process the message (e.g. explicit spam)."""
    return {
        "verdict": "SUSPICIOUS",
        "confidence_score": 0.6,
        "fraud_type": "other",
        "reasons": [
            "Le contenu du message a été bloqué par le filtre de sécurité du fournisseur d'IA "
            "(contenu explicite ou interdit) : il n'a pas pu être analysé automatiquement",
            "Les messages non sollicités au contenu explicite sont un vecteur fréquent de spam et de liens malveillants",
        ],
        "recommended_actions": [
            "Ne pas cliquer sur les liens du message",
            "Supprimer le message s'il n'est pas sollicité et bloquer l'expéditeur",
        ],
        "cited_sources": [p.text[:200] for p in passages[:1]],
    }


async def run_analyze(
    request: AnalyzeRequest,
    settings: dict,
    prompt_dir: Path,
) -> AnalyzeResponse:
    version = settings.get("prompts", {}).get("active_version", "v2")
    prompt_template = _load_prompt(prompt_dir, version)

    context = _format_context(request.passages)
    prompt = prompt_template.replace("{context}", context).replace("{message}", request.extracted_text)

    client: genai.Client = settings["gemini"]["_client"]
    try:
        raw_output, _ = await generate(
            client,
            model_chain(settings, "llm_model"),
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                thinking_config=thinking(settings, "analyze"),
            ),
        )
        data = _extract_json(raw_output)
    except ContentBlocked:
        data = _blocked_verdict(request.passages)
    data = _apply_guardrails(data)

    return AnalyzeResponse(
        extracted_text=request.extracted_text,
        passages=request.passages,
        image_base64=request.image_base64,
        filename=request.filename,
        verdict=data["verdict"],
        confidence_score=data["confidence_score"],
        fraud_type=data["fraud_type"],
        reasons=data["reasons"],
        recommended_actions=data["recommended_actions"],
        cited_sources=data["cited_sources"],
        is_retry=request.is_retry,
    )
