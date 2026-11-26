"""
Retrieval evaluation on the 15 test messages.
Metrics: Hit@5, MRR, Precision@5, Recall@5, False Positive Rate (FPR).

Definitions:
  - Relevant: a retrieved passage whose 'label' matches the query's true label
  - Hit@5: fraction of queries with ≥1 relevant passage in top 5
  - MRR: mean(1/rank_of_first_relevant_passage); 0 if no relevant passage found
  - Precision@5: mean(relevant_in_top5 / 5)
  - Recall@5: mean(relevant_in_top5 / total_relevant_in_corpus) — approximated
               by dividing by the corpus label count
  - FPR: for ham queries, fraction of spam passages in top 5
         (measures false alarm rate on legitimate messages)
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import yaml

# Allow running as a script from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient

from rag.retriever import embed_queries, gemini_client, search


def load_settings() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_test_set(test_dir: Path):
    texts_path = test_dir / "textes.jsonl"
    annots_path = test_dir / "annotations.jsonl"

    texts = {}
    with open(texts_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            texts[item["id"]] = item

    annotations = {}
    with open(annots_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            annotations[item["id"]] = item

    queries = []
    for msg_id, text_item in texts.items():
        ann = annotations.get(msg_id, {})
        queries.append({
            "id": msg_id,
            "text": text_item["text"],
            "source": text_item.get("source", "unknown"),
            "true_label": ann.get("label", "unknown"),
            "fraud_type": ann.get("fraud_type", None),
        })
    return queries


def count_corpus_labels(corpus_dir: Path) -> Dict[str, int]:
    """Count how many corpus documents have each label (for recall denominator)."""
    counts: Dict[str, int] = {}
    for txt_file in corpus_dir.glob("*.txt"):
        content = txt_file.read_text(encoding="utf-8")
        label = "unknown"
        first_line = content.splitlines()[0] if content.splitlines() else ""
        if first_line.startswith("# "):
            for part in first_line[2:].split():
                if part.startswith("label="):
                    label = part.split("=", 1)[1]
        counts[label] = counts.get(label, 0) + 1
    return counts


def evaluate_single(
    query: Dict[str, Any],
    passages: List[Dict[str, Any]],
    top_k: int,
    corpus_label_counts: Dict[str, int],
) -> Dict[str, Any]:
    true_label = query["true_label"]
    relevant_flags = [1 if p["label"] == true_label else 0 for p in passages]

    hit = int(any(relevant_flags))
    n_relevant = sum(relevant_flags)

    # MRR
    mrr = 0.0
    for rank, flag in enumerate(relevant_flags, 1):
        if flag:
            mrr = 1.0 / rank
            break

    precision = n_relevant / top_k if top_k > 0 else 0.0

    # Recall (approximate)
    total_relevant = corpus_label_counts.get(true_label, 1)
    recall = min(n_relevant / total_relevant, 1.0)

    # FPR: only meaningful for ham queries
    fpr = None
    if true_label == "ham":
        spam_in_top_k = sum(1 for p in passages if p["label"] == "spam")
        fpr = spam_in_top_k / top_k if top_k > 0 else 0.0

    return {
        "id": query["id"],
        "true_label": true_label,
        "hit@5": hit,
        "mrr": round(mrr, 4),
        "precision@5": round(precision, 4),
        "recall@5": round(recall, 4),
        "fpr": round(fpr, 4) if fpr is not None else "N/A",
        "top_passages": [{"rank": p["rank"], "label": p["label"], "score": p["score"], "text": p["text"][:80] + "..."} for p in passages],
    }


def run_evaluation(
    strategy: str,
    queries: List[Dict[str, Any]],
    query_vectors: List[List[float]],
    settings: dict,
) -> Dict[str, Any]:
    corpus_dir = Path(__file__).parent.parent / "corpus"
    corpus_label_counts = count_corpus_labels(corpus_dir)
    top_k = settings["rag"]["retrieval"]["top_k"]

    qdrant = QdrantClient(
        host=os.environ.get("QDRANT_HOST", "localhost"),
        port=int(os.environ.get("QDRANT_PORT", settings["qdrant"]["port"])),
    )
    collection = f"{settings['qdrant']['collection']}_{strategy}"

    per_query_results = []
    print(f"\nEvaluating {len(queries)} queries (strategy: {strategy}, collection: {collection}, top_k={top_k})\n")

    for q, vector in zip(queries, query_vectors):
        passages = search(vector, qdrant, collection, top_k=top_k)
        result = evaluate_single(q, passages, top_k, corpus_label_counts)
        per_query_results.append(result)
        label_tag = f"[{q['true_label'].upper()}]"
        print(f"{q['id']} {label_tag:8} Hit={result['hit@5']} MRR={result['mrr']:.3f} P@5={result['precision@5']:.3f}")

    n = len(per_query_results)
    ham_results = [r for r in per_query_results if r["true_label"] == "ham"]

    aggregates = {
        "strategy": strategy,
        "n_queries": n,
        "Hit@5": round(sum(r["hit@5"] for r in per_query_results) / n, 4),
        "MRR": round(sum(r["mrr"] for r in per_query_results) / n, 4),
        "Precision@5": round(sum(r["precision@5"] for r in per_query_results) / n, 4),
        "Recall@5": round(sum(r["recall@5"] for r in per_query_results) / n, 4),
        "FPR (ham queries)": round(
            sum(r["fpr"] for r in ham_results if r["fpr"] != "N/A") / len(ham_results), 4
        ) if ham_results else "N/A",
    }

    print("\n── Aggregate Metrics ──────────────────────────────────")
    for k, v in aggregates.items():
        print(f"  {k}: {v}")

    return {"aggregates": aggregates, "per_query": per_query_results}


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "both"
    strategies = ["fixed", "recursive"] if arg == "both" else [arg]

    settings = load_settings()
    queries = load_test_set(Path(__file__).parent.parent / "data" / "test_messages")
    embed_model = os.environ.get("GEMINI_EMBED_MODEL", settings["gemini"]["embed_model"])
    # Same query vectors for every strategy: one API call, fair comparison.
    query_vectors = embed_queries(gemini_client(), [q["text"] for q in queries], embed_model)

    summary = []
    for strategy in strategies:
        results = run_evaluation(strategy, queries, query_vectors, settings)
        summary.append(results["aggregates"])
        out_path = Path(__file__).parent.parent / "docs" / f"eval_results_{strategy}.json"
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {out_path}")

    if len(summary) > 1:
        print("\n── Comparison ─────────────────────────────────────────")
        keys = ["Hit@5", "MRR", "Precision@5", "Recall@5", "FPR (ham queries)"]
        print(f"  {'strategy':10} " + " ".join(f"{k:>18}" for k in keys))
        for agg in summary:
            print(f"  {agg['strategy']:10} " + " ".join(f"{agg[k]:>18}" for k in keys))
