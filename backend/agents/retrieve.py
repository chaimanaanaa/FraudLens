"""
RAG Agent — embeds the extracted text with Gemini (gemini-embedding-001) and queries Qdrant.
Each chunking strategy has its own collection: <collection>_<strategy>.
"""
import asyncio
from typing import List

import httpx
from google import genai
from google.genai import errors, types
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

from backend.agents.gemini import TRANSIENT_CODES
from backend.models import Passage, RetrieveRequest, RetrieveResponse


async def _embed(client: genai.Client, text: str, model: str, attempts: int = 3) -> List[float]:
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            result = await client.aio.models.embed_content(
                model=model,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
            return list(result.embeddings[0].values)
        except (errors.APIError, httpx.TransportError) as exc:
            transient = isinstance(exc, httpx.TransportError) or exc.code in TRANSIENT_CODES
            if not transient or attempt == attempts:
                raise
            await asyncio.sleep(delay)
            delay *= 2


def _scored_point_to_passage(point: ScoredPoint) -> Passage:
    payload = point.payload or {}
    return Passage(
        text=payload.get("text", ""),
        score=round(point.score, 4),
        source=payload.get("source", "unknown"),
        label=payload.get("label", "unknown"),
    )


async def run_retrieve(
    request: RetrieveRequest,
    settings: dict,
) -> RetrieveResponse:
    strategy = settings["rag"]["active_chunking_strategy"]
    top_k = request.top_k or settings["rag"]["retrieval"]["top_k"]

    if not request.extracted_text.strip():
        return RetrieveResponse(
            extracted_text=request.extracted_text,
            image_base64=request.image_base64,
            filename=request.filename,
            passages=[],
            strategy=strategy,
            top_k=top_k,
        )

    client: genai.Client = settings["gemini"]["_client"]
    query_vector = await _embed(client, request.extracted_text, settings["gemini"]["embed_model"])

    qdrant = QdrantClient(host=settings["qdrant"]["host"], port=settings["qdrant"]["port"])
    results = await asyncio.to_thread(
        qdrant.search,
        collection_name=f"{settings['qdrant']['collection']}_{strategy}",
        query_vector=query_vector,
        limit=top_k,
        score_threshold=settings["rag"]["retrieval"].get("score_threshold", 0.35),
    )

    return RetrieveResponse(
        extracted_text=request.extracted_text,
        image_base64=request.image_base64,
        filename=request.filename,
        passages=[_scored_point_to_passage(r) for r in results],
        strategy=strategy,
        top_k=top_k,
    )
