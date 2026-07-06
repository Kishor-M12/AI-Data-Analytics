# 📊 RAG-Based AI Data Analytics Platform

An intelligent data analytics system using **Retrieval-Augmented Generation (RAG)** that lets you upload datasets and ask natural language questions to get insights, SQL queries, and visualizations.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│              Streamlit Frontend                  │
│  Upload │ Chat │ Charts │ Quality │ History      │
└──────────────────┬──────────────────────────────┘
                   │ HTTP (REST)
┌──────────────────▼──────────────────────────────┐
│              FastAPI Backend                     │
│  /api/upload │ /api/query │ /api/session         │
└──────────────────┬──────────────────────────────┘
                   │
       ┌───────────▼────────────┐
       │      Pipeline          │
       │  Ingest → Embed → RAG  │
       └───────────┬────────────┘
          ┌────────┴──────────┐
          ▼                   ▼
   ┌─────────────┐    ┌──────────────┐
   │ FAISS Index │    │  BM25 Index  │
   │ (Semantic)  │    │  (Keyword)   │
   └──────┬──────┘    └──────┬───────┘
          └────────┬──────────┘
                   ▼
          Reciprocal Rank Fusion
                   │
                   ▼
            LLM (OpenAI / Local)
                   │
          ┌────────┴────────┐
          ▼                 ▼
    Text Insight      Plotly Chart
```

---

## 📁 Project Structure

```
AI_Data/
├── main.py                      # FastAPI app entry point
├── config.py                    # Centralized settings (pydantic-settings)
├── requirements.txt
├── .env.example                 # Environment variables template
├── start.bat                    # Windows: start both servers
│
├── backend/
│   ├── pipeline.py              # ⭐ Main orchestrator
│   ├── ingestion/
│   │   ├── loader.py            # CSV/Excel/JSON/DB loading
│   │   ├── cleaner.py           # Data quality + preprocessing
│   │   └── chunker.py           # DataFrame → text chunks
│   ├── embedding/
│   │   ├── embedder.py          # SentenceTransformers / OpenAI embeddings
│   │   └── vector_store.py      # FAISS index build/save/load/search
│   ├── retrieval/
│   │   └── hybrid_search.py     # ⭐ Hybrid search (FAISS + BM25 + RRF)
│   ├── llm/
│   │   ├── prompt_builder.py    # Structured prompt construction
│   │   └── llm_client.py        # OpenAI / local LLM calls
│   ├── visualization/
│   │   └── chart_engine.py      # Plotly chart auto-generation
│   ├── api/
│   │   └── routes.py            # FastAPI endpoints
│   └── utils/
│       ├── logger.py            # JSON + console logging
│       ├── cache.py             # Disk-based caching (diskcache)
│       └── query_log.py         # Query audit log (JSONL)
│
├── frontend/
│   └── app.py                   # Streamlit UI
│
├── data/
│   ├── generate_sample.py       # Generate sample e-commerce CSV
│   └── sample_ecommerce.csv     # Ready-to-use test dataset
│
├── vector_store/                # FAISS index files (auto-created)
└── logs/                        # App + query logs (auto-created)
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.11+
- pip or UV Package Manager 

### 2. Install Dependencies
```bash
pip install -r requirements.txt  
```
If it shows some Error Install it in line by line or one by one modules 

### 3. Configure Environment
```bash
copy .env.example .env
```
Edit `.env` and set your `OPENAI_API_KEY`.

> **No OpenAI key?** Set `EMBEDDING_PROVIDER=sentence_transformers` and `LLM_PROVIDER=local` to use free local models (requires Ollama).

### 4. Run the Platform

**Windows (recommended):**
```bash
start.bat
```

**Manual (two terminals):**
```bash
# Terminal 1 — Backend
python main.py

# Terminal 2 — Frontend
streamlit run frontend/app.py
```

### 5. Open in Browser
| Service  | URL |
|----------|-----|
| Frontend | http://localhost:8501 |
| API      | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

---

## 🚀 Usage

1. Open **http://localhost:8501**
2. Upload `Your own Data File`
3. Click **Ingest Dataset** — wait for embedding to complete
4. Type a question in the chat, e.g.:
   - `"Top 5 products by revenue"`
   - `"Sales trend over time"`
   - `"Distribution of customer age"`
   - `"Which region has highest sales?"`
5. View the **text insight**, **chart**, **SQL query**, and **retrieved chunks**

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload and ingest a dataset |
| `POST` | `/api/query` | Run a NL query |
| `GET`  | `/api/session/{id}` | Get session info |
| `DELETE` | `/api/session/{id}` | Delete session |
| `GET`  | `/api/history` | Query log history |
| `GET`  | `/api/cache/stats` | Cache statistics |
| `DELETE` | `/api/cache` | Clear cache |
| `GET`  | `/health` | Health check |

---

## 🧠 Module Explanations

| Module | Role |
|--------|------|
| `ingestion/loader.py` | Parses CSV/Excel/JSON/DB into DataFrames |
| `ingestion/cleaner.py` | Handles nulls, duplicates, type inference |
| `ingestion/chunker.py` | Converts DataFrame into RAG-ready text chunks (rows, columns, summary) |
| `embedding/embedder.py` | Generates dense vector embeddings (SentenceTransformers or OpenAI) |
| `embedding/vector_store.py` | FAISS flat index for fast cosine similarity search |
| `retrieval/hybrid_search.py` | **BM25 + FAISS + Reciprocal Rank Fusion** for best retrieval accuracy |
| `llm/prompt_builder.py` | Structures context + query into LLM-ready prompt |
| `llm/llm_client.py` | Calls LLM and parses structured JSON response |
| `visualization/chart_engine.py` | Auto-generates bar/line/histogram/pie/scatter charts from LLM instructions |
| `backend/pipeline.py` | Orchestrates the full ingest and query pipelines |
| `backend/api/routes.py` | FastAPI REST endpoints |
| `frontend/app.py` | Streamlit chat UI with charts, quality reports, history |
| `utils/cache.py` | Disk-based LRU cache for embeddings and LLM responses |
| `utils/query_log.py` | JSONL query audit trail |

---

## 🔧 Configuration Options (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required for OpenAI provider |
| `LLM_PROVIDER` | `openai` | `openai` or `local` |
| `EMBEDDING_PROVIDER` | `sentence_transformers` | `sentence_transformers` or `openai` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model name |
| `BACKEND_PORT` | `8000` | FastAPI server port |
| `CACHE_TTL` | `3600` | Cache TTL in seconds |

---

## 🏆 Hybrid Search — Why It Matters

```
Query: "ORD-1042 revenue"        → BM25 wins   (exact ID match)
Query: "top selling items"       → FAISS wins  (semantic match)
Query: "January sales by region" → Both win    (date keyword + meaning)
```

RRF formula: `score(d) = Σ 1 / (60 + rank_i)` across ranked lists

This ensures neither pure keyword nor pure semantic search dominates — the best results from both always surface.
=======
# AI-Data-Analytics
