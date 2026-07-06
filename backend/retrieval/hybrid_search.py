"""
hybrid_search.py — Hybrid retrieval combining semantic search (FAISS) and
keyword search (BM25) using Reciprocal Rank Fusion (RRF).

Why hybrid?
  - Semantic search handles meaning-based / paraphrased queries
  - BM25 handles exact keyword matches (IDs, dates, product names)
  - RRF merges ranked lists without requiring score normalization
"""
import numpy as np
from rank_bm25 import BM25Okapi
from backend.embedding.vector_store import FAISSVectorStore
from backend.embedding.embedder import embed_query
from backend.utils.logger import get_logger

logger = get_logger(__name__)

RRF_K = 60  # RRF constant — controls influence of rank position


class HybridSearchEngine:
    """
    Combines FAISS semantic search and BM25 keyword search.

    Usage:
        engine = HybridSearchEngine(vector_store)
        engine.build_bm25(chunks)
        results = engine.search("top 5 products by revenue", top_k=5)
    """

    def __init__(self, vector_store: FAISSVectorStore):
        self.vector_store = vector_store
        self.bm25: BM25Okapi | None = None
        self.bm25_chunks: list[dict] = []

    def build_bm25(self, chunks: list[dict]) -> None:
        """
        Build BM25 index from chunk texts.

        Args:
            chunks: List of chunk dicts with 'text' field
        """
        self.bm25_chunks = chunks
        tokenized = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built: {len(chunks)} documents")

    def search(self, query: str, top_k: int = 10, semantic_weight: float = 0.6) -> list[dict]:
        """
        Run hybrid search and return RRF-fused ranked results.

        Args:
            query: Natural language query string
            top_k: Number of final results to return
            semantic_weight: Not used in RRF but logged for context
        Returns:
            Ranked list of chunk dicts with 'hybrid_score' and 'retrieval_source'
        """
        semantic_results = self._semantic_search(query, top_k=top_k * 2)
        keyword_results = self._bm25_search(query, top_k=top_k * 2)
        fused = self._reciprocal_rank_fusion(semantic_results, keyword_results, top_k=top_k)
        logger.info(f"Hybrid search for '{query[:60]}': {len(fused)} results returned")
        return fused

    def _semantic_search(self, query: str, top_k: int) -> list[dict]:
        """FAISS cosine similarity search."""
        q_vec = embed_query(query)
        return self.vector_store.search(q_vec, top_k=top_k)

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """BM25 keyword search over chunk texts."""
        if self.bm25 is None:
            return []
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = dict(self.bm25_chunks[idx])
                chunk["bm25_score"] = float(scores[idx])
                results.append(chunk)
        return results

    def _reciprocal_rank_fusion(
        self,
        semantic: list[dict],
        keyword: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        RRF score = Σ 1 / (k + rank_i) for each list
        Higher score = better combined rank.
        """
        scores: dict[str, float] = {}
        chunk_map: dict[str, dict] = {}

        for rank, chunk in enumerate(semantic):
            cid = chunk["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            chunk_map[cid] = chunk

        for rank, chunk in enumerate(keyword):
            cid = chunk["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = chunk

        ranked_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
        results = []
        for cid in ranked_ids:
            chunk = dict(chunk_map[cid])
            chunk["hybrid_score"] = round(scores[cid], 6)
            # Tag source
            in_semantic = any(c["id"] == cid for c in semantic)
            in_keyword = any(c["id"] == cid for c in keyword)
            chunk["retrieval_source"] = (
                "both" if in_semantic and in_keyword
                else "semantic" if in_semantic
                else "keyword"
            )
            results.append(chunk)
        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenizer for BM25."""
        return text.lower().split()
