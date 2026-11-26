"""
Chunking Strategy 2 — Recursive / semantic chunking.
Standalone implementation (no langchain dependency) using a hierarchy of separators.

Rationale: Splits on paragraph → sentence → word boundaries, preserving semantic
units. Better for fraud examples because each example tends to be a self-contained
paragraph that should not be cut mid-thought.

Weakness: Chunk sizes vary more; harder to guarantee token-count limits.
"""
from typing import List, Dict, Any


def _split_text(text: str, separators: List[str]) -> List[str]:
    """
    Recursively split text using the first separator that works,
    then re-merge small pieces and re-split large ones.
    Mirrors LangChain's RecursiveCharacterTextSplitter logic without the dependency.
    """
    if not separators:
        return [text]
    sep = separators[0]
    if sep == "":
        return list(text)
    return text.split(sep)


def _merge_splits(splits: List[str], separator: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Merge small splits back into chunks of at most chunk_size characters with overlap."""
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    sep_len = len(separator)

    for split in splits:
        split_len = len(split)
        projected = current_len + split_len + (sep_len if current else 0)

        if projected > chunk_size and current:
            # Flush current chunk
            chunk = separator.join(current).strip()
            if chunk:
                chunks.append(chunk)
            # Slide window: keep the last overlap-worth of splits
            while current and current_len > chunk_overlap:
                removed = current.pop(0)
                current_len -= len(removed) + (sep_len if current else 0)

        current.append(split)
        current_len += split_len + (sep_len if len(current) > 1 else 0)

    if current:
        chunk = separator.join(current).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: List[str],
) -> List[str]:
    """
    Try separators in order. Split by the first one that creates pieces
    small enough, then recursively split any piece that's still too large.
    """
    if len(text) <= chunk_size:
        return [text]

    for i, sep in enumerate(separators):
        remaining_seps = separators[i + 1:]
        parts = text.split(sep) if sep else list(text)

        good: List[str] = []
        for part in parts:
            if not part.strip():
                continue
            if len(part) <= chunk_size:
                good.append(part)
            else:
                # Too big — recurse with remaining separators
                good.extend(_recursive_split(part, chunk_size, chunk_overlap, remaining_seps))

        if len(good) > 1 or (len(good) == 1 and len(good[0]) <= chunk_size):
            return _merge_splits(good, sep, chunk_size, chunk_overlap)

    # Last resort: character-level truncation
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]


def chunk_recursive(
    text: str,
    metadata: Dict[str, Any],
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    separators: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Split `text` recursively using paragraph → sentence → word boundaries.

    Args:
        text: Raw document text.
        metadata: Payload to attach to every chunk.
        chunk_size: Target max characters per chunk (not tokens — character-based).
        chunk_overlap: Overlap in characters between consecutive chunks.
        separators: Ordered list of separators to try.

    Returns:
        List of dicts with keys: text, chunk_index, total_chunks, strategy, + metadata fields.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    raw_chunks = _recursive_split(text, chunk_size, chunk_overlap, separators)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        stripped = chunk_text.strip()
        if stripped:
            chunks.append({
                "text": stripped,
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
                "strategy": "recursive",
                "chunk_size_chars": len(stripped),
                **metadata,
            })

    return chunks


def chunk_documents_recursive(
    documents: List[Dict[str, Any]],
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    separators: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Apply recursive chunking to a list of documents.
    Each document must have a "text" key and arbitrary metadata.
    """
    all_chunks = []
    for doc in documents:
        text = doc.pop("text", "")
        chunks = chunk_recursive(
            text, metadata=doc,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
        all_chunks.extend(chunks)
    return all_chunks
