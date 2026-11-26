"""
End-to-end evaluation of the agent chain on the 15 test messages (text input, so the
Vision Agent is skipped): /retrieve -> /analyze -> /validate on the running backend.

Metrics:
  - detection: spam counts as detected if the verdict is SCAM or SUSPICIOUS
  - accuracy, precision, recall, F1, false positive rate (ham flagged)
  - first-pass validation rate (validation passed without regeneration)

Usage:
  python scripts/evaluate_pipeline.py [backend_url]     # default http://localhost:8000
"""
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
BACKEND = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
PAUSE_SECONDS = 4  # stay under the free-tier requests-per-minute limit


def load_jsonl(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return {item["id"]: item for item in map(json.loads, f)}


def post(route: str, payload: dict) -> dict:
    response = requests.post(f"{BACKEND}/{route}", json=payload, timeout=300)
    if response.status_code != 200:
        raise RuntimeError(f"/{route} -> {response.status_code}: {response.text[:300]}")
    return response.json()


def run_one(text: str) -> dict:
    retrieved = post("retrieve", {"extracted_text": text, "top_k": 5})
    analysis = post("analyze", {"extracted_text": text, "passages": retrieved["passages"]})
    return post("validate", analysis)


def main() -> None:
    texts = load_jsonl(ROOT / "data" / "test_messages" / "textes.jsonl")
    labels = load_jsonl(ROOT / "data" / "test_messages" / "annotations.jsonl")

    rows = []
    for i, (msg_id, item) in enumerate(texts.items()):
        if i:
            time.sleep(PAUSE_SECONDS)
        true_label = labels[msg_id]["label"]
        start = time.time()
        try:
            result = run_one(item["text"])
            row = {
                "id": msg_id,
                "true_label": true_label,
                "verdict": result["verdict"],
                "confidence": result["confidence_score"],
                "fraud_type": result.get("fraud_type"),
                "is_valid": result["is_valid"],
                "regenerated": result["regenerated"],
                "validation_issues": result["validation_issues"],
                "n_passages": len(result["passages"]),
                "latency_s": round(time.time() - start, 1),
            }
        except Exception as exc:
            row = {"id": msg_id, "true_label": true_label, "error": str(exc)}
        rows.append(row)
        print(f"{msg_id} [{true_label:4}] -> {row.get('verdict', 'ERROR'):10} "
              f"regenerated={row.get('regenerated', '-')} {row.get('latency_s', '')}s"
              + (f"  {row['error'][:120]}" if "error" in row else ""))

    ok = [r for r in rows if "error" not in r]
    tp = sum(1 for r in ok if r["true_label"] == "spam" and r["verdict"] != "LEGITIMATE")
    fn = sum(1 for r in ok if r["true_label"] == "spam" and r["verdict"] == "LEGITIMATE")
    fp = sum(1 for r in ok if r["true_label"] == "ham" and r["verdict"] != "LEGITIMATE")
    tn = sum(1 for r in ok if r["true_label"] == "ham" and r["verdict"] == "LEGITIMATE")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    summary = {
        "n_messages": len(rows),
        "n_errors": len(rows) - len(ok),
        "accuracy": round((tp + tn) / len(ok), 3) if ok else 0.0,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0,
        "false_positive_rate": round(fp / (fp + tn), 3) if fp + tn else 0.0,
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "verdicts": {v: sum(1 for r in ok if r["verdict"] == v) for v in ("SCAM", "SUSPICIOUS", "LEGITIMATE")},
        "first_pass_validation_rate": round(sum(1 for r in ok if not r["regenerated"]) / len(ok), 3) if ok else 0.0,
        "final_valid_rate": round(sum(1 for r in ok if r["is_valid"]) / len(ok), 3) if ok else 0.0,
        "mean_latency_s": round(sum(r["latency_s"] for r in ok) / len(ok), 1) if ok else 0.0,
    }

    print("\n── Summary ────────────────────────────────────────────")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    out = ROOT / "docs" / "pipeline_eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_message": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
