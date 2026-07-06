"""
vector_store.py — FAISS-based vector store for semantic search.

Stores chunk embeddings with their metadata. Supports:
  - Building index from embedded chunks
  - Persisting index to disk
  - Loading from disk
  - Top-K similarity search
"""
import os
import pickle
import numpy as np
import faiss
from config import get_settings
from backend.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class FAISSVectorStore:
    """Wraps a FAISS flat index with chunk metadata storage."""

    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None  # Inner-product (cosine on normalized vecs)
        self.chunks: list[dict] = []                  # Parallel list of chunk metadata + text
        self.dim: int = 0

    def build(self, chunks: list[dict]) -> None:
        """
        Build FAISS index from a list of embedded chunks.

        Args:
            chunks: List of dicts each containing 'embedding' (np.ndarray)
        """
        embeddings = np.array([c["embedding"] for c in chunks], dtype=np.float32)
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)   # cosine sim via normalized vectors
        self.index.add(embeddings)
        # Store chunks without the embedding array (saves memory)
        self.chunks = [{k: v for k, v in c.items() if k != "embedding"} for c in chunks]
        logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={self.dim}")

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[dict]:
        """
        Retrieve top-K most similar chunks.

        Args:
            query_vec: 1D normalized query embedding
            top_k: Number of results
        Returns:
            List of chunk dicts with added 'score' field
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        q = query_vec.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(q, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def save(self, name: str | None = None) -> None:
        """Persist FAISS index and chunk metadata to disk."""
        name = name or settings.faiss_index_name
        os.makedirs(settings.vector_store_path, exist_ok=True)
        faiss.write_index(self.index, self._index_path(name))
        with open(self._meta_path(name), "wb") as f:
            pickle.dump({"chunks": self.chunks, "dim": self.dim}, f)
        logger.info(f"Saved FAISS index '{name}'")

    def load(self, name: str | None = None) -> bool:
        """
        Load persisted FAISS index from disk.

        Returns:
            True if loaded successfully, False if not found
        """
        name = name or settings.faiss_index_name
        if not os.path.exists(self._index_path(name)):
            return False
        self.index = faiss.read_index(self._index_path(name))
        with open(self._meta_path(name), "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.dim = data["dim"]
        logger.info(f"Loaded FAISS index '{name}': {self.index.ntotal} vectors")
        return True

    def _index_path(self, name: str) -> str:
        return os.path.join(settings.vector_store_path, f"{name}.faiss")

    def _meta_path(self, name: str) -> str:
        return os.path.join(settings.vector_store_path, f"{name}.pkl")
