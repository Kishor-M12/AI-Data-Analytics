"""
routes.py — FastAPI route definitions for the RAG analytics API.

Endpoints:
  POST /api/upload          → Ingest a dataset file
  POST /api/query           → Run a natural language query
  GET  /api/session/{id}    → Get session metadata
  DELETE /api/session/{id}  → Reset/delete a session
  GET  /api/history         → Retrieve query log history
  GET  /api/cache/stats     → Cache statistics
  DELETE /api/cache         → Clear cache
  GET  /health              → Health check
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.pipeline import ingest, query, get_session_info, reset_session
from backend.utils.query_log import get_query_history
from backend.utils.cache import cache_stats, clear_cache
from backend.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ── Request / Response Models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    session_id: str
    query: str
    top_k: int = 8


class QueryResponse(BaseModel):
    text_insight: str
    sql_query: str | None
    chart_type: str
    chart: dict | None
    chart_config: dict
    retrieved_chunks: list[dict]
    confidence: float
    response_time_ms: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/api/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    session_id: str | None = None,
):
    """
    Upload and ingest a CSV / Excel / JSON file.
    Returns session_id and data quality report.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        result = ingest(file_bytes, file.filename, session_id=session_id)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")


@router.post("/api/query", response_model=QueryResponse)
def run_query(req: QueryRequest):
    """
    Run a natural language query against an ingested dataset.
    Returns LLM insight, SQL, chart config, and retrieved chunks.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = query(req.session_id, req.query, top_k=req.top_k)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@router.get("/api/session/{session_id}")
def session_info(session_id: str):
    """Get metadata for an active session."""
    try:
        return get_session_info(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    """Remove a session from memory."""
    reset_session(session_id)
    return {"message": f"Session {session_id} deleted"}


@router.get("/api/history")
def query_history(limit: int = Query(default=50, ge=1, le=500)):
    """Return recent query log entries."""
    return {"history": get_query_history(limit=limit)}


@router.get("/api/cache/stats")
def get_cache_stats():
    return cache_stats()


@router.delete("/api/cache")
def clear_all_cache():
    clear_cache()
    return {"message": "Cache cleared"}
