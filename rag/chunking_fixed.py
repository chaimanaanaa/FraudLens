"""
Chunking Strategy 1 — Fixed-size token chunking.
Splits text into chunks of a fixed token count with overlap.

Rationale: Simple, reproducible, fast. Good baseline for comparison.
Weakness: Can cut mid-sentence, breaking semantic units in fraud examples.
"""
from typing import List, Dict, Any
import tiktoken


def chunk_fixed(
    text: str,
    metadata: Dict[str, Any],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    encoding_name: str = "cl100k_base",
) -> List[Dict[str, Any]]:
    """
    Split `text` into fixed-size token chunks with overlap.

    Args:
        text: Raw document text.
        metadata: Payload to attach to every chunk (source, label, etc.).
        chunk_size: Max tokens per chunk.
        chunk_overlap: Number of overlapping tokens between consecutive chunks.
        encoding_name: tiktoken encoding (cl100k_base covers GPT-3.5/4 and is a
                       good proxy for token counts in other models).

    Returns:
        List of dicts with keys: text, chunk_index, total_chunks, strategy, + metadata fields.
    """
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens).strip()

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "chunk_index": chunk_index,
                "strategy": "fixed",
                "chunk_size_tokens": len(chunk_tokens),
                **metadata,
            })
            chunk_index += 1

        if end >= len(tokens):
            break
        start = end - chunk_overlap

    # Backfill total_chunks now that we know the count
    for c in chunks:
        c["total_chunks"] = chunk_index

    return chunks


def chunk_documents_fixed(
    documents: List[Dict[str, Any]],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Apply fixed-size chunking to a list of documents.
    Each document must have a "text" key and arbitrary metadata.
    """
    all_chunks = []
    for doc in documents:
        text = doc.pop("text", "")
        chunks = chunk_fixed(text, metadata=doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)
    return all_chunks
