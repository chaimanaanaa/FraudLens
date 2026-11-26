"""
Shared Gemini call helper.
  - 5xx: retried with exponential backoff, then the next model is tried
  - 404 (model retired), 429 (quota, often per model and per day), network error or timeout: next model immediately
  - input blocked by the provider's safety filter: ContentBlocked (no other model would accept it)
"""
import asyncio
import logging
from typing import Any, List, Optional, Tuple

import httpx
from google import genai
from google.genai import errors, types

log = logging.getLogger("fraudlens.gemini")

TRANSIENT_CODES = {429, 500, 502, 503, 504}
NEXT_MODEL_CODES = {404, 429}


class ContentBlocked(ValueError):
    """The provider refused the input (e.g. PROHIBITED_CONTENT). Not retryable."""


def model_chain(settings: dict, primary_key: str) -> List[str]:
    primary = settings["gemini"][primary_key]
    fallbacks = settings["gemini"].get("fallback_models", [])
    return [primary] + [m for m in fallbacks if m != primary]


def thinking(settings: dict, agent: str) -> Optional[types.ThinkingConfig]:
    level = (settings["gemini"].get("thinking_level") or {}).get(agent)
    return types.ThinkingConfig(thinking_level=level) if level else None


def _blocked_reason(response: types.GenerateContentResponse) -> Optional[str]:
    if response.prompt_feedback and response.prompt_feedback.block_reason:
        return str(response.prompt_feedback.block_reason)
    if response.candidates and response.candidates[0].finish_reason in (
        types.FinishReason.SAFETY, types.FinishReason.PROHIBITED_CONTENT,
        types.FinishReason.BLOCKLIST, types.FinishReason.SPII,
        types.FinishReason.IMAGE_SAFETY, types.FinishReason.IMAGE_PROHIBITED_CONTENT,
    ):
        return str(response.candidates[0].finish_reason)
    return None


async def generate(
    client: genai.Client,
    models: List[str],
    contents: Any,
    config: types.GenerateContentConfig,
    attempts_per_model: int = 2,
) -> Tuple[str, str]:
    """Returns (response_text, model_used). Raises ContentBlocked or RuntimeError."""
    last_exc: Exception | None = None
    for model in models:
        delay = 2.0
        for attempt in range(1, attempts_per_model + 1):
            try:
                response = await client.aio.models.generate_content(
                    model=model, contents=contents, config=config
                )
            except httpx.TransportError as exc:
                last_exc = exc
                log.warning("Model %s network error (%s) — trying next model", model, type(exc).__name__)
                break
            except errors.APIError as exc:
                last_exc = exc
                if exc.code in NEXT_MODEL_CODES:
                    log.warning("Model %s returned %s — trying next model", model, exc.code)
                    break
                if exc.code not in TRANSIENT_CODES:
                    raise
                log.warning("Model %s returned %s (attempt %d/%d)", model, exc.code, attempt, attempts_per_model)
                if attempt < attempts_per_model:
                    await asyncio.sleep(delay)
                    delay *= 2
                continue

            blocked = _blocked_reason(response)
            if blocked:
                raise ContentBlocked(f"Input blocked by Gemini safety filter ({blocked})")
            if response.text:
                return response.text, model
            last_exc = RuntimeError(f"empty response from {model}")
            log.warning("Model %s returned an empty response — trying next model", model)
            break
    raise RuntimeError(f"All Gemini models failed ({', '.join(models)}). Last error: {last_exc}")
