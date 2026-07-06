"""
query_log.py — Persistent query logging for audit trail and analytics.
Appends each query + response metadata to a JSONL log file.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from config import get_settings

settings = get_settings()
QUERY_LOG_FILE = os.path.join(settings.log_dir, "queries.jsonl")


def log_query(
    query: str,
    session_id: str,
    retrieved_chunks: int,
    confidence: float,
    response_time_ms: float,
    chart_type: Optional[str] = None,
) -> None:
    """
    Append a query record to the JSONL query log.

    Args:
        query: The user's natural language query
        session_id: Unique session identifier
        retrieved_chunks: Number of chunks retrieved from vector store
        confidence: Confidence score (0-1)
        response_time_ms: Time taken to generate response
        chart_type: Type of chart generated, if any
    """
    Path(settings.log_dir).mkdir(exist_ok=True)
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "confidence": round(confidence, 4),
        "response_time_ms": round(response_time_ms, 2),
        "chart_type": chart_type,
    }
    with open(QUERY_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def get_query_history(limit: int = 50) -> list[dict]:
    """Return the last N query log entries."""
    if not os.path.exists(QUERY_LOG_FILE):
        return []
    with open(QUERY_LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    records = [json.loads(l) for l in lines if l.strip()]
    return records[-limit:]



