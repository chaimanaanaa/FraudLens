"""
Standalone retriever — used outside the FastAPI backend (e.g. in evaluate.py).
Embeds queries with Gemini (task_type RETRIEVAL_QUERY) and searches Qdrant.
"""
import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()


def gemini_client(api_key: Optional[str] = None) -> genai.Client:
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set. Add it to your .env file.")
    return genai.Client(api_key=key)


def embed_queries(client: genai.Client, texts: List[str], model: str) -> List[List[float]]:
    """Embeds several queries in a single API call."""
    result = client.models.embed_content(
        model=model,
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return [list(e.values) for e in result.embeddings]


def search(
    query_vector: List[float],
    qdrant: QdrantClient,
    collection: str,
    top_k: int = 5,
    score_threshold: float = 0.0,
    label_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    qdrant_filter = None
    if label_filter:
        qdrant_filter = Filter(must=[FieldCondition(key="label", match=MatchValue(value=label_filter))])

    results = qdrant.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
        query_filter=qdrant_filter,
    )
    return [
        {
            "text": r.payload.get("text", ""),
            "score": round(r.score, 4),
            "source": r.payload.get("source", "unknown"),
            "label": r.payload.get("label", "unknown"),
            "filename": r.payload.get("filename", ""),
            "rank": i + 1,
        }
        for i, r in enumerate(results)
    ]


def retrieve(
    query: str,
    qdrant_host: str,
    qdrant_port: int,
    collection: str,
    embed_model: str,
    top_k: int = 5,
    score_threshold: float = 0.0,
    label_filter: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    vector = embed_queries(gemini_client(api_key), [query], embed_model)[0]
    qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
    return search(vector, qdrant, collection, top_k, score_threshold, label_filter)
