"""
Runs prompts v1 and v2 on the same cases with the same model.
Writes prompts/v1_v2_results.json after each call, and resumes from it if interrupted.

Needs GEMINI_API_KEY and the running backend (retrieval context).
Uses 20 generation requests.

Usage: python scripts/compare_prompts.py
"""
import base64
import difflib
import io
import json
import os
import sys
import time
from pathlib import Path

import httpx
import requests
import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.agents.analyze import REFUND_KEYWORDS, _extract_json, _format_context  # noqa: E402
from backend.models import Passage  # noqa: E402

BACKEND = "http://localhost:8000"
OUTPUT = ROOT / "prompts" / "v1_v2_results.json"
PAUSE_SECONDS = 13  # stay under the free-tier requests-per-minute limit

SMS_LINES = [
    "LA POSTE: Votre colis n'a pas pu etre livre.",
    "Des frais de 1,99 EUR sont a regler sous 24h",
    "sinon il sera retourne a l'expediteur :",
    "http://laposte-suivi-colis.info/pay",
]

ANALYZE_CASES = {
    "scam_parcel": " ".join(SMS_LINES),
    "legit_appointment": "Bonjour, nous vous rappelons votre rendez-vous du jeudi 14 novembre a 10h30 "
                         "au cabinet du Dr Martin. En cas d'empechement, merci d'appeler le 01 42 00 00 00.",
    "ambiguous_family": "Salut maman, j'ai change de numero. Ecris-moi sur WhatsApp, c'est important.",
    "lottery_victim": "Felicitations ! Votre gain de 5000 EUR est bloque. Vous avez deja regle 300 EUR : "
                      "versez 50 EUR de plus sur http://gain-rapide.xyz pour le debloquer.",
}


def png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def sms_image() -> Image.Image:
    image = Image.new("RGB", (720, 240), "#E5E5EA")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=28)
    for index, line in enumerate(SMS_LINES):
        draw.text((30, 30 + 44 * index), line, fill="black", font=font)
    return image


def no_text_image() -> Image.Image:
    image = Image.new("RGB", (720, 240), "#DCE7F5")
    ImageDraw.Draw(image).ellipse((280, 40, 440, 200), fill="#1D4ED8")
    return image


IMAGE_CASES = {
    "clear_sms": sms_image(),
    "blurred_sms": sms_image().filter(ImageFilter.GaussianBlur(3.2)),
    "no_text": no_text_image(),
}


def load_results() -> dict:
    return json.loads(OUTPUT.read_text(encoding="utf-8")) if OUTPUT.exists() else {"runs": {}}


class Gemini:
    def __init__(self, models: list):
        self.client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(timeout=120000, retry_options=types.HttpRetryOptions(attempts=1)),
        )
        self.models = list(models)

    def call(self, contents, config: types.GenerateContentConfig):
        """Returns (text, model). Network errors are retried; a model that keeps failing is dropped."""
        network_errors = 0
        while self.models:
            model = self.models[0]
            try:
                response = self.client.models.generate_content(model=model, contents=contents, config=config)
                return response.text or "", model
            except httpx.TransportError as exc:
                network_errors += 1
                print(f"  {model} network error ({type(exc).__name__}), attempt {network_errors}", flush=True)
                if network_errors >= 3:
                    self.models.pop(0)
                    network_errors = 0
                time.sleep(20)
            except errors.APIError as exc:
                print(f"  {model} failed ({exc.code}), switching model", flush=True)
                self.models.pop(0)
                time.sleep(5)
        raise RuntimeError("no model available")


def extract_metrics(raw: str, case: str) -> dict:
    truth = "\n".join(SMS_LINES)
    return {
        "similarity_to_truth": None if case == "no_text" else round(difflib.SequenceMatcher(None, truth, raw.strip()).ratio(), 3),
        "no_text_sentinel": "NO_TEXT_FOUND" in raw.upper(),
        "uncertainty_marks": raw.count("[?]"),
    }


def analyze_metrics(raw: str) -> dict:
    try:
        data = _extract_json(raw)
    except ValueError:
        return {"parsed": False, "markdown_fence": "```" in raw}
    actions = data.get("recommended_actions") or []
    return {
        "parsed": True,
        "markdown_fence": "```" in raw,
        "verdict": data.get("verdict"),
        "confidence": data.get("confidence_score"),
        "reasons": len(data.get("reasons") or []),
        "actions": len(actions),
        "cited_sources": len(data.get("cited_sources") or []),
        "refund_promise": any(kw in str(a).lower() for a in actions for kw in REFUND_KEYWORDS),
    }


def validate_metrics(raw: str) -> dict:
    try:
        data = _extract_json(raw)
    except ValueError:
        return {"parsed": False}
    return {"parsed": True, "all_checks_passed": data.get("all_checks_passed"), "checks": data.get("checks"), "issues": data.get("issues")}


def validation_cases(passages: list) -> dict:
    base = {
        "verdict": "SCAM",
        "confidence_score": 0.9,
        "reasons": ["Domaine non officiel", "Urgence artificielle sous 24h"],
        "recommended_actions": ["Ne pas cliquer sur le lien", "Signaler le SMS au 33700"],
    }
    return {
        "valid_analysis": {**base, "cited_sources": [passages[0]["text"][:120]]},
        "invented_citation": {**base, "cited_sources": ["Arnaque au faux remboursement de la CAF signalee en 2024"]},
        "refund_and_full_confidence": {
            **base,
            "confidence_score": 1.0,
            "recommended_actions": ["Contactez votre banque pour obtenir le remboursement de vos frais"],
            "cited_sources": [passages[0]["text"][:120]],
        },
    }


def main() -> None:
    load_dotenv(ROOT / ".env")
    settings = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))["gemini"]
    gemini = Gemini([settings["llm_model"], *settings["fallback_models"]])
    results = load_results()
    prompts = {
        (agent, version): (ROOT / "prompts" / version / f"{agent}.txt").read_text(encoding="utf-8").strip()
        for agent in ("extract", "analyze", "validate")
        for version in ("v1", "v2")
    }

    passages = {
        case: requests.post(f"{BACKEND}/retrieve", json={"extracted_text": text, "top_k": 5}, timeout=60).json()["passages"]
        for case, text in ANALYZE_CASES.items()
    }

    def run(key: str, contents, config, metrics) -> None:
        if key in results["runs"]:
            return
        raw, model = gemini.call(contents, config)
        results["runs"][key] = {"model": model, "metrics": metrics(raw), "raw": raw[:1200]}
        OUTPUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(key, model, results["runs"][key]["metrics"], flush=True)
        time.sleep(PAUSE_SECONDS)

    plain = types.GenerateContentConfig(temperature=0.1)
    as_json = types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json")

    for case, image in IMAGE_CASES.items():
        part = types.Part.from_bytes(data=png(image), mime_type="image/png")
        for version in ("v1", "v2"):
            run(f"extract/{case}/{version}", [prompts["extract", version], part],
                types.GenerateContentConfig(temperature=0.0), lambda raw, c=case: extract_metrics(raw, c))

    for case, text in ANALYZE_CASES.items():
        context = _format_context([Passage(**p) for p in passages[case]])
        for version, config in (("v1", plain), ("v2", as_json)):
            prompt = prompts["analyze", version].replace("{context}", context).replace("{message}", text)
            run(f"analyze/{case}/{version}", prompt, config, analyze_metrics)

    knowledge = "\n".join(f"- {p['text']}" for p in passages["scam_parcel"])
    for case, analysis in validation_cases(passages["scam_parcel"]).items():
        payload = json.dumps(analysis, ensure_ascii=False, indent=2)
        for version, config in (("v1", plain), ("v2", as_json)):
            prompt = prompts["validate", version].replace("{analysis}", payload).replace("{passages}", knowledge)
            run(f"validate/{case}/{version}", prompt, config, validate_metrics)

    print(f"\nDone: {len(results['runs'])} runs saved to {OUTPUT}")


if __name__ == "__main__":
    main()
