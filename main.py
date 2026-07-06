"""
main.py — FastAPI application entry point.

Registers:
  - CORS middleware (allows Streamlit frontend to call the API)
  - All API routes from backend/api/routes.py
  - Startup/shutdown lifecycle events
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from contextlib import asynccontextmanager
import os
from pathlib import Path

from backend.api.routes import router
from backend.utils.logger import get_logger
from config import get_settings

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create required directories on startup."""
    for d in (settings.vector_store_path, settings.log_dir, settings.cache_dir):
        Path(d).mkdir(parents=True, exist_ok=True)
    logger.info("RAG Data Analytics started")
    yield
    logger.info("RAG Data Analytics shutting down")


app = FastAPI(
    title="RAG Data Analytics",
    description="Intelligent data analytics using Retrieval-Augmented Generation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/info")
def info():
    """API info and available endpoints."""
    return JSONResponse({
        "name": "RAG Data Analytics API",
        "version": "1.0.0",
        "status": "running",
        "docs": "http://localhost:8000/docs",
        "health": "http://localhost:8000/health",
        "endpoints": [
            "POST /api/upload",
            "POST /api/query",
            "GET  /api/session/{id}",
            "DELETE /api/session/{id}",
            "GET  /api/history",
            "GET  /api/cache/stats",
            "DELETE /api/cache",
            "GET  /health",
            "GET  /info",
        ]
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=False,
    )
