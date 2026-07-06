"""
llm_client.py — LLM integration for generating structured analytics responses.

Supports:
  - OpenAI GPT models (gpt-4o, gpt-3.5-turbo)
  - Local models via LiteLLM-compatible interface

Always expects a structured JSON response as defined in prompt_builder.py.
"""
import json
import re
from backend.llm.prompt_builder import build_prompt
from backend.utils.logger import get_logger
from backend.utils.cache import cached
from config import get_settings

settings = get_settings()
logger = get_logger(__name__)


def query_llm(query: str, chunks: list[dict]) -> dict:
    """
    Send query + context to LLM and return parsed structured response.

    Args:
        query: User's natural language question
        chunks: Retrieved context chunks from hybrid search
    Returns:
        Parsed dict with keys: text_insight, sql_query, chart_type,
        chart_config, confidence
    """
    messages = build_prompt(query, chunks)
    raw = _call_llm(messages)
    return _parse_response(raw)


@cached(ttl=1800)  # Cache identical prompt responses for 30 min
def _call_llm(messages: tuple) -> str:
    """
    Call the configured LLM provider.

    Args:
        messages: Tuple of message dicts (tuple for cache hashability)
    Returns:
        Raw string response from the model
    """
    provider = settings.llm_provider.lower()

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=list(messages),
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    elif provider == "local":
        # LiteLLM or Ollama compatible endpoint
        import httpx
        payload = {
            "model": settings.local_model_path,
            "messages": list(messages),
            "temperature": 0.2,
        }
        resp = httpx.post("http://localhost:11434/api/chat", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    raise ValueError(f"Unknown LLM provider: {provider}")


def _parse_response(raw: str) -> dict:
    """
    Parse LLM response string into a structured dict.
    Falls back gracefully if JSON is malformed.
    """
    # Strip markdown code fences if present
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON response, wrapping as text_insight")
        data = {
            "text_insight": raw,
            "sql_query": None,
            "chart_type": "none",
            "chart_config": {"x": None, "y": None, "group_by": None, "title": "", "agg": None},
            "confidence": 0.4,
        }

    # Ensure all expected keys exist
    data.setdefault("sql_query", None)
    data.setdefault("chart_type", "none")
    data.setdefault("chart_config", {})
    data.setdefault("confidence", 0.5)
    data["chart_config"].setdefault("x", None)
    data["chart_config"].setdefault("y", None)
    data["chart_config"].setdefault("group_by", None)
    data["chart_config"].setdefault("title", "")
    data["chart_config"].setdefault("agg", None)

    return data
