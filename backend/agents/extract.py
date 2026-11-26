"""
Vision Agent — calls Gemini (multimodal) to extract text from an image.
Uses prompt v2/extract.txt (role prompting + NO_TEXT_FOUND sentinel).
"""
import base64
import io
import re
from pathlib import Path

import PIL.Image
from google import genai
from google.genai import types

from backend.agents.gemini import generate, model_chain, thinking
from backend.models import ExtractRequest, ExtractResponse


def _load_prompt(prompt_dir: Path, version: str) -> str:
    return (prompt_dir / version / "extract.txt").read_text(encoding="utf-8").strip()


def _clean_base64(image_b64: str) -> str:
    if "," in image_b64:
        return image_b64.split(",", 1)[1]
    return image_b64


def _strip_narrative(raw: str) -> str:
    patterns = [
        r"^(the image (shows|contains|displays|reads)[:\s]*)",
        r"^(this (image|picture|photo|message|text) (shows|contains|says|reads)[:\s]*)",
        r"^(i can see[:\s]*)",
        r"^(the text (in the image )?reads[:\s]*)",
        r"^(extracted text[:\s]*)",
    ]
    result = raw.strip()
    for pat in patterns:
        result = re.sub(pat, "", result, flags=re.IGNORECASE).strip()
    return result


def _image_part(image_b64: str) -> types.Part:
    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
        img = PIL.Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise ValueError(f"image_base64 is not a valid image: {exc}") from exc
    fmt = (img.format or "JPEG").lower()
    mime_type = "image/jpeg" if fmt in ("jpg", "jpeg", "mpo") else f"image/{fmt}"
    return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)


async def run_extract(
    request: ExtractRequest,
    settings: dict,
    prompt_dir: Path,
) -> ExtractResponse:
    version = settings.get("prompts", {}).get("active_version", "v2")
    prompt = _load_prompt(prompt_dir, version)
    client: genai.Client = settings["gemini"]["_client"]

    image = _image_part(_clean_base64(request.image_base64))
    raw_output, model_used = await generate(
        client,
        model_chain(settings, "vision_model"),
        contents=[prompt, image],
        config=types.GenerateContentConfig(temperature=0.0, thinking_config=thinking(settings, "extract")),
    )

    if "NO_TEXT_FOUND" in raw_output.upper():
        extracted = ""
    else:
        extracted = _strip_narrative(raw_output)

    return ExtractResponse(
        image_base64=request.image_base64,
        filename=request.filename,
        extracted_text=extracted,
        raw_model_output=raw_output,
        model_used=model_used,
    )
