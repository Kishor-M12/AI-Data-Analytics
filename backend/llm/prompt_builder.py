"""
prompt_builder.py — Construct LLM prompts from query and retrieved context chunks.

The prompt instructs the LLM to return a structured JSON response containing:
  - text_insight: Natural language explanation
  - sql_query: SQL-like query if applicable
  - chart_type: Suggested visualization type
  - chart_config: Axes and grouping instructions for the chart
  - confidence: Self-assessed confidence score (0.0 - 1.0)
"""

SYSTEM_PROMPT = """You are an expert data analyst AI. You receive:
1. A user question about a dataset
2. Relevant data context chunks retrieved from that dataset

Your job is to analyze the context and answer accurately.

Always respond in this exact JSON format:
{
  "text_insight": "<detailed natural language answer>",
  "sql_query": "<SQL SELECT query if applicable, else null>",
  "chart_type": "<one of: bar, line, histogram, pie, scatter, none>",
  "chart_config": {
    "x": "<column name for x-axis or null>",
    "y": "<column name for y-axis or null>",
    "group_by": "<column name to group/color by or null>",
    "title": "<chart title>",
    "agg": "<aggregation: sum, mean, count, or null>"
  },
  "confidence": <float 0.0 to 1.0>
}

Rules:
- Base your answer ONLY on the provided context
- If data is insufficient, set confidence < 0.5 and explain in text_insight
- Choose chart_type = "none" if visualization doesn't make sense
- Keep sql_query standard ANSI SQL compatible
"""


def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    """
    Build a chat messages list for the LLM API.

    Args:
        query: User's natural language question
        chunks: Retrieved chunks from hybrid search
    Returns:
        List of role/content message dicts
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("retrieval_source", "unknown")
        score = chunk.get("hybrid_score", chunk.get("score", 0))
        context_parts.append(
            f"[Chunk {i} | source={source} | score={score:.4f}]\n{chunk['text']}"
        )

    context_text = "\n\n---\n\n".join(context_parts)

    user_message = f"""Question: {query}

Retrieved Context:
{context_text}

Answer in the required JSON format."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
