"""
Validator Agent — checks that the Analysis output meets all guardrails.

Hybrid validation:
  1. Deterministic code checks (authoritative for confidence, refund, reasons, actions).
  2. LLM judge (prompt v2/validate.txt) for the semantic check the code cannot do:
     are the cited sources actually grounded in the retrieved passages?
     If the LLM call fails, the code checks alone decide (the pipeline never breaks here).
If validation fails, run_analyze is called once (is_retry=True) and its result is accepted.
"""
import json
import logging
from pathlib import Path
from typing import List, Optional

from google import genai
from google.genai import types

from backend.agents.analyze import _extract_json, run_analyze
from backend.agents.gemini import generate, model_chain, thinking
from backend.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    Passage,
    ValidateRequest,
    ValidateResponse,
    ValidationChecks,
)

log = logging.getLogger("fraudlens.validate")

REFUND_KEYWORDS = [
    "remboursement", "rembourser", "récupérer votre argent",
    "refund", "reimbursement", "recover your money",
]


def _load_prompt(prompt_dir: Path, version: str) -> str:
    return (prompt_dir / version / "validate.txt").read_text(encoding="utf-8").strip()


def _format_passages(passages: List[Passage]) -> str:
    if not passages:
        return "No passages retrieved."
    return "\n".join(f"- {p.text}" for p in passages)


def _run_code_checks(request: ValidateRequest) -> ValidationChecks:
    return ValidationChecks(
        confidence_below_max=request.confidence_score < 1.0,
        sources_cited=len(request.cited_sources) > 0,
        no_refund_promise=not any(
            kw in action.lower()
            for action in request.recommended_actions
            for kw in REFUND_KEYWORDS
        ),
        reasons_sufficient=len(request.reasons) >= 2,
        actions_present=len(request.recommended_actions) >= 1,
    )


async def _llm_sources_grounded(
    request: ValidateRequest, settings: dict, prompt_dir: Path
) -> Optional[bool]:
    """Returns the LLM verdict on citation grounding, or None if the LLM is unavailable."""
    version = settings.get("prompts", {}).get("active_version", "v2")
    analysis = {
        "verdict": request.verdict,
        "confidence_score": request.confidence_score,
        "reasons": request.reasons,
        "recommended_actions": request.recommended_actions,
        "cited_sources": request.cited_sources,
    }
    prompt = (
        _load_prompt(prompt_dir, version)
        .replace("{analysis}", json.dumps(analysis, ensure_ascii=False, indent=2))
        .replace("{passages}", _format_passages(request.passages))
    )
    client: genai.Client = settings["gemini"]["_client"]
    try:
        raw, _ = await generate(
            client,
            model_chain(settings, "llm_model"),
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                thinking_config=thinking(settings, "validate"),
            ),
        )
        return bool(_extract_json(raw).get("checks", {}).get("sources_cited", True))
    except Exception as exc:
        log.warning("LLM validation unavailable, using code checks only: %s", exc)
        return None


def _collect_issues(checks: ValidationChecks) -> List[str]:
    issues = []
    if not checks.confidence_below_max:
        issues.append("confidence_score must be < 1.0 (got 1.0 or higher)")
    if not checks.sources_cited:
        issues.append("cited_sources must contain at least one excerpt grounded in the retrieved passages")
    if not checks.no_refund_promise:
        issues.append("recommended_actions must not promise refund or money recovery")
    if not checks.reasons_sufficient:
        issues.append("reasons must contain at least 2 items")
    if not checks.actions_present:
        issues.append("recommended_actions must not be empty")
    return issues


async def _validate(request: ValidateRequest, settings: dict, prompt_dir: Path) -> ValidationChecks:
    checks = _run_code_checks(request)
    if checks.sources_cited and request.passages:
        grounded = await _llm_sources_grounded(request, settings, prompt_dir)
        if grounded is False:
            checks.sources_cited = False
    return checks


async def run_validate(
    request: ValidateRequest,
    settings: dict,
    prompt_dir: Path,
) -> ValidateResponse:
    checks = await _validate(request, settings, prompt_dir)
    issues = _collect_issues(checks)

    if not issues:
        return ValidateResponse(
            is_valid=True,
            validation_issues=[],
            regenerated=False,
            checks=checks,
            verdict=request.verdict,
            confidence_score=request.confidence_score,
            fraud_type=request.fraud_type,
            reasons=request.reasons,
            recommended_actions=request.recommended_actions,
            cited_sources=request.cited_sources,
            extracted_text=request.extracted_text,
            passages=request.passages,
        )

    # Validation failed — regenerate the analysis exactly once and accept it.
    retry_result: AnalyzeResponse = await run_analyze(
        AnalyzeRequest(
            extracted_text=request.extracted_text,
            passages=request.passages,
            image_base64=request.image_base64,
            filename=request.filename,
            is_retry=True,
        ),
        settings,
        prompt_dir,
    )
    retry_checks = _run_code_checks(ValidateRequest(
        extracted_text=retry_result.extracted_text,
        passages=retry_result.passages,
        verdict=retry_result.verdict,
        confidence_score=retry_result.confidence_score,
        fraud_type=retry_result.fraud_type,
        reasons=retry_result.reasons,
        recommended_actions=retry_result.recommended_actions,
        cited_sources=retry_result.cited_sources,
        is_retry=True,
    ))

    return ValidateResponse(
        is_valid=not _collect_issues(retry_checks),
        validation_issues=issues,
        regenerated=True,
        checks=retry_checks,
        verdict=retry_result.verdict,
        confidence_score=retry_result.confidence_score,
        fraud_type=retry_result.fraud_type,
        reasons=retry_result.reasons,
        recommended_actions=retry_result.recommended_actions,
        cited_sources=retry_result.cited_sources,
        extracted_text=request.extracted_text,
        passages=request.passages,
    )
