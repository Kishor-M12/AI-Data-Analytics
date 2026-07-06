"""
pipeline.py — Central orchestrator for the RAG analytics pipeline.

Flow:
  1. ingest()  → Load file, clean data, chunk text, embed, build FAISS + BM25
  2. query()   → Embed query, hybrid search, LLM call, chart generation
  3. reset()   → Clear current session state

SessionState holds per-session data so multiple datasets can coexist.
"""
import time
import uuid
import pandas as pd
from dataclasses import dataclass, field

from backend.ingestion.loader import load_file
from backend.ingestion.cleaner import clean_dataframe, get_data_summary
from backend.ingestion.chunker import chunk_dataframe
from backend.embedding.embedder import embed_chunks
from backend.embedding.vector_store import FAISSVectorStore
from backend.retrieval.hybrid_search import HybridSearchEngine
from backend.llm.llm_client import query_llm
from backend.visualization.chart_engine import generate_chart
from backend.utils.logger import get_logger
from backend.utils.query_log import log_query

logger = get_logger(__name__)


@dataclass
class SessionState:
    """Holds all state for a single uploaded dataset session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    df: pd.DataFrame | None = None
    quality_report: dict = field(default_factory=dict)
    data_summary: dict = field(default_factory=dict)
    chunks: list[dict] = field(default_factory=list)
    vector_store: FAISSVectorStore = field(default_factory=FAISSVectorStore)
    search_engine: HybridSearchEngine | None = None
    dataset_name: str = ""
    ready: bool = False


# Global in-memory session store  {session_id: SessionState}
_sessions: dict[str, SessionState] = {}


def ingest(file_bytes: bytes, filename: str, session_id: str | None = None) -> dict:
    """
    Full ingestion pipeline: load → clean → chunk → embed → index.

    Args:
        file_bytes: Raw uploaded file bytes
        filename: Original filename
        session_id: Reuse existing session ID or create new one
    Returns:
        Dict with session_id, quality_report, data_summary, chunk_count
    """
    state = SessionState()
    if session_id:
        state.session_id = session_id

    # 1. Load
    state.df = load_file(file_bytes, filename)
    state.dataset_name = filename

    # 2. Clean
    state.df, state.quality_report = clean_dataframe(state.df)
    state.data_summary = get_data_summary(state.df)

    # 3. Chunk
    state.chunks = chunk_dataframe(state.df, dataset_name=filename)

    # 4. Embed
    state.chunks = embed_chunks(state.chunks)

    # 5. Build FAISS index
    state.vector_store.build(state.chunks)
    state.vector_store.save(state.session_id)

    # 6. Build BM25 index
    state.search_engine = HybridSearchEngine(state.vector_store)
    state.search_engine.build_bm25(state.chunks)

    state.ready = True
    _sessions[state.session_id] = state

    logger.info(f"Session {state.session_id} ingested: {len(state.chunks)} chunks")

    return {
        "session_id": state.session_id,
        "rows": state.df.shape[0],
        "columns": state.df.shape[1],
        "chunk_count": len(state.chunks),
        "quality_report": state.quality_report,
        "data_summary": state.data_summary,
    }


def query(session_id: str, user_query: str, top_k: int = 8) -> dict:
    """
    Full query pipeline: retrieve → LLM → chart.

    Args:
        session_id: Active session ID from ingest()
        user_query: Natural language question
        top_k: Number of context chunks to retrieve
    Returns:
        Dict with text_insight, sql_query, chart_type, chart (plotly dict),
        retrieved_chunks, confidence, response_time_ms
    """
    state = _get_session(session_id)
    t0 = time.perf_counter()

    # 1. Hybrid search
    chunks = state.search_engine.search(user_query, top_k=top_k)

    # 2. LLM call
    llm_result = query_llm(user_query, chunks)

    # 3. Chart generation
    chart = None
    if llm_result["chart_type"] != "none" and state.df is not None:
        chart = generate_chart(state.df, llm_result["chart_config"], llm_result["chart_type"])

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # 4. Log query
    log_query(
        query=user_query,
        session_id=session_id,
        retrieved_chunks=len(chunks),
        confidence=llm_result["confidence"],
        response_time_ms=elapsed_ms,
        chart_type=llm_result.get("chart_type"),
    )

    return {
        "text_insight": llm_result["text_insight"],
        "sql_query": llm_result["sql_query"],
        "chart_type": llm_result["chart_type"],
        "chart": chart,
        "chart_config": llm_result["chart_config"],
        "retrieved_chunks": [
            {"text": c["text"], "score": c.get("hybrid_score", 0), "source": c.get("retrieval_source")}
            for c in chunks
        ],
        "confidence": llm_result["confidence"],
        "response_time_ms": round(elapsed_ms, 2),
    }


def get_session_info(session_id: str) -> dict:
    """Return metadata for an active session."""
    state = _get_session(session_id)
    return {
        "session_id": state.session_id,
        "dataset_name": state.dataset_name,
        "rows": state.df.shape[0] if state.df is not None else 0,
        "columns": state.df.shape[1] if state.df is not None else 0,
        "chunk_count": len(state.chunks),
        "ready": state.ready,
    }


def reset_session(session_id: str) -> None:
    """Remove a session from memory."""
    _sessions.pop(session_id, None)
    logger.info(f"Session {session_id} reset")


def _get_session(session_id: str) -> SessionState:
    state = _sessions.get(session_id)
    if not state or not state.ready:
        raise ValueError(f"Session '{session_id}' not found or not ready. Please upload a dataset first.")
    return state
