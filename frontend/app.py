"""
app.py — Streamlit frontend for the RAG Data Analytics Platform.

Features:
  - File upload (CSV / Excel / JSON)
  - Data quality report display
  - Chat interface with query history
  - Interactive Plotly chart rendering
  - Retrieved chunks explainability panel
  - Confidence score display
  - Session management
"""
import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

API_BASE = "http://localhost:8000"
st.set_page_config(
    page_title="RAG Analytics Platform",
    page_icon="MK_WEB.jpg",
    layout="wide",
    initial_sidebar_state="expanded",
)

for key, default in [
    ("session_id", None),
    ("chat_history", []),
    ("data_summary", {}),
    ("quality_report", {}),
    ("dataset_info", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Upload needs a long timeout — first run downloads the embedding model (~90MB)
UPLOAD_TIMEOUT = 600   # 10 min
QUERY_TIMEOUT  = 180   # 3 min


def api_post(endpoint: str, timeout: int = QUERY_TIMEOUT, **kwargs) -> dict | None:
    try:
        resp = requests.post(f"{API_BASE}{endpoint}", **kwargs, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Make sure uvicorn is running: `uvicorn main:app --port 8000`")
    except requests.exceptions.ReadTimeout:
        st.error("⏱ Request timed out. The backend is still processing — wait a moment and try again. "
                 "On first run, the embedding model (~90MB) needs to download.")
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"API Error: {detail}")
    return None


def api_get(endpoint: str, timeout: int = 60) -> dict | None:
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(" Cannot connect to backend.")
    except requests.exceptions.ReadTimeout:
        st.error("⏱ Request timed out.")
    except Exception as e:
        st.error(f" {e}")
    return None


def render_chart(chart_dict: dict) -> None:
    """Render a Plotly figure dict in Streamlit."""
    try:
        fig = go.Figure(chart_dict)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Chart render error: {e}")


def confidence_badge(score: float) -> str:
    if score >= 0.75:
        return f"🟢 High ({score:.0%})"
    elif score >= 0.5:
        return f"🟡 Medium ({score:.0%})"
    return f"🔴 Low ({score:.0%})"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("RAG Based Data Analytics")
    st.markdown("---")

    # File Upload
    st.subheader("First Upload Dataset")
    uploaded_file = st.file_uploader(
        "CSV / Excel / JSON",
        type=["csv", "xlsx", "xls", "json"],
        help="Max recommended size: 50MB",
    )

    if uploaded_file and st.button("Ingest Dataset", type="primary"):
        with st.spinner("Ingesting… first run downloads the embedding model, please wait up to 5 min"):
            result = api_post(
                "/api/upload",
                timeout=UPLOAD_TIMEOUT,
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                params={"session_id": st.session_state.session_id} if st.session_state.session_id else {},
            )
        if result:
            st.session_state.session_id = result["session_id"]
            st.session_state.quality_report = result.get("quality_report", {})
            st.session_state.data_summary = result.get("data_summary", {})
            st.session_state.dataset_info = {
                "name": uploaded_file.name,
                "rows": result["rows"],
                "columns": result["columns"],
                "chunks": result["chunk_count"],
            }
            st.session_state.chat_history = []
            st.success(f"✅ Ingested! {result['rows']} rows · {result['columns']} cols · {result['chunk_count']} chunks")

    # Session Info
    if st.session_state.session_id:
        st.markdown("---")
        st.subheader("Active Session")
        info = st.session_state.dataset_info
        st.markdown(f"**File:** {info.get('name', 'N/A')}")
        st.markdown(f"**Rows:** {info.get('rows', 0):,}  |  **Cols:** {info.get('columns', 0)}")
        st.markdown(f"**Chunks:** {info.get('chunks', 0)}")
        st.markdown(f"**Session ID:** `{st.session_state.session_id}`")

        if st.button(" Reset Session"):
            api_get(f"/api/session/{st.session_state.session_id}")
            requests.delete(f"{API_BASE}/api/session/{st.session_state.session_id}", timeout=10)
            for key in ("session_id", "chat_history", "data_summary", "quality_report", "dataset_info"):
                st.session_state[key] = [] if key == "chat_history" else (None if key == "session_id" else {})
            st.rerun()

    # History Tab
    st.markdown("---")
    st.subheader(" Query Log")
    if st.button("Load History"):
        hist = api_get("/api/history?limit=20")
        if hist:
            df_hist = pd.DataFrame(hist.get("history", []))
            if not df_hist.empty:
                st.dataframe(df_hist[["timestamp", "query", "confidence", "response_time_ms"]].tail(10), use_container_width=True)
            else:
                st.info("No queries logged yet.")


#Main
st.title("RAG Data Analytics ")

# Tabs
tab_chat, tab_quality, tab_summary = st.tabs(["💬 Chat", "🔍 Data Quality", "📈 Data Summary"])

with tab_chat:
    if not st.session_state.session_id:
        st.info("👈 Upload a dataset in the sidebar to get started.")
    else:
        # Render chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                else:
                    # Assistant message
                    st.markdown(msg["content"])
                    if msg.get("chart"):
                        render_chart(msg["chart"])
                    if msg.get("sql"):
                        with st.expander("🗃️ Generated SQL"):
                            st.code(msg["sql"], language="sql")
                    if msg.get("chunks"):
                        with st.expander(f"🔎 Retrieved Chunks ({len(msg['chunks'])}) — Explainability"):
                            for i, chunk in enumerate(msg["chunks"], 1):
                                src_icon = {"semantic": "🔵", "keyword": "🟠", "both": "🟢"}.get(chunk.get("source", ""), "⚪")
                                st.markdown(
                                    f"**Chunk {i}** {src_icon} `{chunk.get('source', 'N/A')}` | score: `{chunk.get('score', 0):.4f}`"
                                )
                                st.text(chunk["text"][:400] + ("…" if len(chunk["text"]) > 400 else ""))
                    if msg.get("confidence") is not None:
                        st.caption(f"Confidence: {confidence_badge(msg['confidence'])}  ·  ⏱ {msg.get('rt', 0):.0f}ms")

        # Query input
        user_input = st.chat_input("Ask about your data… e.g. 'Top 5 products by revenue'")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    result = api_post(
                        "/api/query",
                        json={"session_id": st.session_state.session_id, "query": user_input, "top_k": 8},
                    )

                if result:
                    # Display response
                    st.markdown(result["text_insight"])
                    if result.get("chart"):
                        render_chart(result["chart"])
                    if result.get("sql_query"):
                        with st.expander("🗃️ Generated SQL"):
                            st.code(result["sql_query"], language="sql")
                    if result.get("retrieved_chunks"):
                        with st.expander(f"🔎 Retrieved Chunks ({len(result['retrieved_chunks'])}) — Explainability"):
                            for i, chunk in enumerate(result["retrieved_chunks"], 1):
                                src_icon = {"semantic": "🔵", "keyword": "🟠", "both": "🟢"}.get(chunk.get("source", ""), "⚪")
                                st.markdown(
                                    f"**Chunk {i}** {src_icon} `{chunk.get('source', 'N/A')}` | score: `{chunk.get('score', 0):.4f}`"
                                )
                                st.text(chunk["text"][:400] + ("…" if len(chunk["text"]) > 400 else ""))
                    st.caption(
                        f"Confidence: {confidence_badge(result['confidence'])}  ·  ⏱ {result['response_time_ms']:.0f}ms"
                    )

                    # Save to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result["text_insight"],
                        "chart": result.get("chart"),
                        "sql": result.get("sql_query"),
                        "chunks": result.get("retrieved_chunks", []),
                        "confidence": result.get("confidence"),
                        "rt": result.get("response_time_ms", 0),
                    })

        # Sample queries
        st.markdown("---")
        st.markdown("**💡 Sample Queries:**")
        samples = [
            "Show top 5 products by revenue",
            "What is the sales trend over time?",
            "Distribution of customer age",
            "Which region has the highest sales?",
            "Show monthly revenue by category",
        ]
        cols = st.columns(len(samples))
        for col, sample in zip(cols, samples):
            if col.button(sample, key=f"sample_{sample[:15]}"):
                st.session_state["_prefill"] = sample
                st.rerun()

        # Handle prefilled sample
        if "_prefill" in st.session_state:
            pf = st.session_state.pop("_prefill")
            st.session_state.chat_history.append({"role": "user", "content": pf})
            with st.spinner("Thinking…"):
                result = api_post(
                    "/api/query",
                    json={"session_id": st.session_state.session_id, "query": pf, "top_k": 8},
                )
            if result:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["text_insight"],
                    "chart": result.get("chart"),
                    "sql": result.get("sql_query"),
                    "chunks": result.get("retrieved_chunks", []),
                    "confidence": result.get("confidence"),
                    "rt": result.get("response_time_ms", 0),
                })
            st.rerun()

with tab_quality:
    if not st.session_state.quality_report:
        st.info("Upload a dataset to view the data quality report.")
    else:
        qr = st.session_state.quality_report
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Original Shape", f"{qr.get('original_shape', ('?','?'))[0]} × {qr.get('original_shape', ('?','?'))[1]}")
        col2.metric("Final Shape", f"{qr.get('final_shape', ('?','?'))[0]} × {qr.get('final_shape', ('?','?'))[1]}")
        col3.metric("Duplicates Removed", qr.get("duplicates_removed", 0))
        col4.metric("Columns with Nulls", len(qr.get("missing_values", {})))

        if qr.get("missing_values"):
            st.subheader("Missing Values")
            mv_df = pd.DataFrame({
                "Column": list(qr["missing_values"].keys()),
                "Missing Count": list(qr["missing_values"].values()),
                "Missing %": [qr["missing_pct"].get(c, 0) for c in qr["missing_values"]],
            })
            st.dataframe(mv_df, use_container_width=True)

        if qr.get("column_types"):
            st.subheader("Column Data Types")
            ct_df = pd.DataFrame(
                {"Column": list(qr["column_types"].keys()), "Type": list(qr["column_types"].values())}
            )
            st.dataframe(ct_df, use_container_width=True)

#text summary
with tab_summary:
    if not st.session_state.data_summary:
        st.info("Upload a dataset to view the data summary.")
    else:
        ds = st.session_state.data_summary

        st.subheader("Numeric Statistics")
        if ds.get("numeric_stats"):
            try:
                stats_df = pd.DataFrame(ds["numeric_stats"]).T
                st.dataframe(stats_df.round(2), use_container_width=True)
            except Exception:
                st.json(ds["numeric_stats"])

        st.subheader("Top Categorical Values")
        if ds.get("categorical_top"):
            for col, counts in ds["categorical_top"].items():
                with st.expander(f"📂 {col}"):
                    cat_df = pd.DataFrame({"Value": list(counts.keys()), "Count": list(counts.values())})
                    st.dataframe(cat_df, use_container_width=True)
