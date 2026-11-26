"""
RAG ingestion pipeline — Gemini gemini-embedding-001 + batching + cache + exponential backoff.
Each chunking strategy is written to its own collection: <collection>_<strategy>.

Stratégie anti-quota (free tier: ~15 req/min, 1500 req/jour):
  - Batching : 50 textes par requête → ~7 appels pour 311 chunks (au lieu de 311)
  - Cache local : data/embeddings_cache.json — survit aux quotas et aux redémarrages
  - Exponential backoff sur 429 : 4 tentatives max (60s → 120s → 240s), pas de boucle infinie
"""
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from rag.chunking_fixed import chunk_documents_fixed
from rag.chunking_recursive import chunk_documents_recursive

load_dotenv()

BATCH_SIZE = 50          # textes par appel API Gemini
MAX_RETRIES = 4          # tentatives max sur 429 (pas de boucle infinie)
BASE_WAIT = 60           # secondes pour la 1re attente (double à chaque tentative)
CACHE_FILE = Path(__file__).parent.parent / "data" / "embeddings_cache.json"


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_settings() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _cache_key(model: str, text: str) -> str:
    """Clé unique par modèle + contenu — résiste aux changements de modèle."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    return f"{model}:{digest}"


def _load_cache() -> Dict[str, List[float]]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(cache: Dict[str, List[float]]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")


def load_corpus(corpus_dir: Path) -> List[Dict[str, Any]]:
    documents = []
    for txt_file in sorted(corpus_dir.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8").strip()
        if not text:
            continue
        label, source = "unknown", txt_file.stem
        lines = text.splitlines()
        if lines[0].startswith("# "):
            for part in lines[0][2:].split():
                if "=" in part:
                    k, v = part.split("=", 1)
                    if k == "label":
                        label = v
                    elif k == "source":
                        source = v
            text = "\n".join(lines[1:]).strip()
        documents.append({"text": text, "source": source, "label": label, "filename": txt_file.name})
    return documents


# ── Embedding avec batching, cache et backoff ──────────────────────────────────

def _embed_batch_with_backoff(
    client: genai.Client,
    texts: List[str],
    model: str,
) -> List[List[float]]:
    """
    Envoie un batch de textes à l'API Gemini.
    Réessaie jusqu'à MAX_RETRIES fois sur 429, avec attente exponentielle.
    Lève RuntimeError si toutes les tentatives échouent (pas de boucle infinie).
    """
    wait = BASE_WAIT
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = client.models.embed_content(
                model=model,
                contents=texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            return [list(emb.values) for emb in result.embeddings]

        except ClientError as e:
            if e.code != 429:
                raise
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"429 Rate Limit après {MAX_RETRIES} tentatives. "
                    f"Les embeddings déjà calculés sont sauvegardés dans {CACHE_FILE}. "
                    f"Relance le script dans quelques minutes pour reprendre."
                ) from e
            print(f"  [429] Tentative {attempt}/{MAX_RETRIES} — attente {wait}s (backoff exponentiel)...")
            time.sleep(wait)
            wait *= 2  # 60s → 120s → 240s


def embed_chunks(
    client: genai.Client,
    chunks: List[Dict[str, Any]],
    model_name: str,
) -> List[Dict[str, Any]]:
    """
    Embed tous les chunks :
      1. Charge le cache local (data/embeddings_cache.json)
      2. Filtre les chunks déjà cachés
      3. Envoie les chunks restants par batches de BATCH_SIZE
      4. Sauvegarde le cache après chaque batch
    """
    cache = _load_cache()

    # Séparer les chunks déjà cachés de ceux à calculer
    to_embed: List[Tuple[int, str]] = []   # (index_in_chunks, text)
    for i, chunk in enumerate(chunks):
        key = _cache_key(model_name, chunk["text"])
        if key not in cache:
            to_embed.append((i, chunk["text"]))

    cached_count = len(chunks) - len(to_embed)
    if cached_count:
        print(f"  Cache : {cached_count}/{len(chunks)} chunks déjà calculés — ignorés.")

    if to_embed:
        total_batches = (len(to_embed) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  {len(to_embed)} chunks à embeder en {total_batches} batch(es) de {BATCH_SIZE}...")

        for b_start in range(0, len(to_embed), BATCH_SIZE):
            batch = to_embed[b_start : b_start + BATCH_SIZE]
            indices, texts = zip(*batch)
            batch_num = b_start // BATCH_SIZE + 1

            print(f"  Batch {batch_num}/{total_batches} ({len(texts)} textes)...", end=" ", flush=True)
            embeddings = _embed_batch_with_backoff(client, list(texts), model_name)
            print("OK")

            # Mettre en cache et sauvegarder après chaque batch
            for idx, emb in zip(indices, embeddings):
                cache[_cache_key(model_name, chunks[idx]["text"])] = emb
            _save_cache(cache)

    # Appliquer les embeddings depuis le cache
    for chunk in chunks:
        chunk["embedding"] = cache[_cache_key(model_name, chunk["text"])]

    return chunks


# ── Qdrant ────────────────────────────────────────────────────────────────────

def collection_name(settings: dict, strategy: str) -> str:
    return f"{settings['qdrant']['collection']}_{strategy}"


def upsert_to_qdrant(
    chunks: List[Dict[str, Any]],
    client: QdrantClient,
    collection: str,
    vector_size: int,
) -> int:
    # Recreate on every run so re-ingesting never duplicates points.
    client.recreate_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Collection Qdrant '{collection}' (re)créée (dim={vector_size})")

    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{collection}/{chunk['filename']}/{chunk['chunk_index']}")),
            vector=chunk["embedding"],
            payload={k: v for k, v in chunk.items() if k != "embedding"},
        )
        for chunk in chunks
    ]
    for i in range(0, len(points), 64):
        client.upsert(collection_name=collection, points=points[i : i + 64])
    return len(points)


# ── Point d'entrée ────────────────────────────────────────────────────────────

def run_ingestion(strategy: str | None = None) -> None:
    settings = load_settings()
    strategy = strategy or settings["rag"]["active_chunking_strategy"]

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY non définie. Ajoute-la dans ton fichier .env.")

    gemini_client = genai.Client(api_key=api_key)

    corpus_dir = Path(__file__).parent.parent / "corpus"
    documents = load_corpus(corpus_dir)
    print(f"Chargé {len(documents)} documents depuis {corpus_dir}")

    if strategy == "fixed":
        cfg = settings["rag"]["fixed"]
        chunks = chunk_documents_fixed(
            documents, chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"]
        )
    else:
        cfg = settings["rag"]["recursive"]
        chunks = chunk_documents_recursive(
            documents,
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            separators=cfg.get("separators"),
        )
    print(f"Produit {len(chunks)} chunks (stratégie '{strategy}')")

    embed_model = os.environ.get("GEMINI_EMBED_MODEL", settings["gemini"]["embed_model"])
    print(f"Modèle d'embedding : {embed_model}")
    chunks = embed_chunks(gemini_client, chunks, embed_model)
    print(f"Embeddings prêts ({len(chunks)} chunks)")

    qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_port = int(os.environ.get("QDRANT_PORT", settings["qdrant"]["port"]))
    qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
    collection = collection_name(settings, strategy)
    n = upsert_to_qdrant(chunks, qdrant, collection, settings["qdrant"]["vector_size"])
    print(f"Upsert de {n} points dans '{collection}'. Terminé.")


if __name__ == "__main__":
    import sys
    run_ingestion(sys.argv[1] if len(sys.argv) > 1 else None)
