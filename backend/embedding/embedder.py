"""
embedder.py — Generate vector embeddings for text chunks.

Supports two providers:
  - sentence_transformers: Local, free, runs on CPU/GPU
  - openai: OpenAI text-embedding API (requires API key)

Embeddings are cached to avoid redundant computation.
"""
import numpy as np
from backend.utils.logger import get_logger
from backend.utils.cache import cached
from config import get_settings

settings = get_settings()
logger = get_logger(__name__)

_st_model = None


def _get_st_model():
    """Lazy-load SentenceTransformer model on first call."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading SentenceTransformer: {settings.embedding_model}")
        _st_model = SentenceTransformer(settings.embedding_model)
    return _st_model


@cached(ttl=86400)
def embed_texts(texts: tuple) -> np.ndarray:
    """
    Embed a tuple of strings into vectors.

    Args:
        texts: Tuple of strings (tuple used for cache key hashability)
    Returns:
        np.ndarray shape (N, dim)
    """
    provider = settings.embedding_provider.lower()

    if provider == "sentence_transformers":
        model = _get_st_model()
        vecs = model.encode(list(texts), show_progress_bar=False, normalize_embeddings=True)
        return np.array(vecs, dtype=np.float32)

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.embeddings.create(input=list(texts), model=settings.openai_embedding_model)
        return np.array([item.embedding for item in resp.data], dtype=np.float32)

    raise ValueError(f"Unknown embedding provider: {provider}")


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string → 1D array."""
    return embed_texts((query,))[0]


def embed_chunks(chunks: list) -> list:
    """
    Attach 'embedding' field to each chunk dict.

    Args:
        chunks: List of dicts with 'text' key
    Returns:
        Same list with 'embedding' np.ndarray added to each dict
    """
    texts = tuple(c["text"] for c in chunks)
    vectors = embed_texts(texts)
    for chunk, vec in zip(chunks, vectors):
        chunk["embedding"] = vec
    logger.info(f"Embedded {len(chunks)} chunks, dim={vectors.shape[1]}")
    return chunks
