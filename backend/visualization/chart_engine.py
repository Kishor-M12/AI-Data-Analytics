"""
chart_engine.py — Auto-generate Plotly charts from LLM chart instructions + DataFrame.

Supported chart types: bar, line, histogram, pie, scatter
Chart type is decided by the LLM based on query intent.
Returns a Plotly figure as a JSON-serializable dict for frontend rendering.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Map LLM chart_type strings to handler functions
CHART_HANDLERS = {}


def register(chart_type: str):
    """Decorator to register a chart builder function."""
    def decorator(fn):
        CHART_HANDLERS[chart_type] = fn
        return fn
    return decorator


def generate_chart(df: pd.DataFrame, chart_config: dict, chart_type: str) -> dict | None:
    """
    Entry point — dispatch to the correct chart builder.

    Args:
        df: The full cleaned DataFrame
        chart_config: Dict with keys: x, y, group_by, title, agg
        chart_type: One of bar, line, histogram, pie, scatter, none
    Returns:
        Plotly figure as JSON dict, or None if chart_type is 'none'
    """
    if chart_type == "none" or not chart_type:
        return None

    handler = CHART_HANDLERS.get(chart_type)
    if handler is None:
        logger.warning(f"No handler for chart_type='{chart_type}', skipping chart")
        return None

    try:
        df_plot = _prepare_data(df, chart_config)
        fig = handler(df_plot, chart_config)
        fig.update_layout(
            template="plotly_white",
            title=chart_config.get("title", ""),
            margin=dict(l=40, r=40, t=50, b=40),
        )
        return fig.to_dict()
    except Exception as e:
        logger.error(f"Chart generation failed for type='{chart_type}': {e}")
        return None


def _prepare_data(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply aggregation if specified in chart_config."""
    x, y, group_by, agg = cfg.get("x"), cfg.get("y"), cfg.get("group_by"), cfg.get("agg")

    if not x or not y or not agg:
        return df

    if agg not in ("sum", "mean", "count"):
        return df

    group_cols = [x] + ([group_by] if group_by and group_by in df.columns else [])
    valid_cols = [c for c in group_cols + [y] if c in df.columns]
    if len(valid_cols) < 2:
        return df

    agg_fn = {"sum": "sum", "mean": "mean", "count": "count"}[agg]
    return df[valid_cols].groupby(group_cols, as_index=False).agg({y: agg_fn})


# ── Chart Builders ────────────────────────────────────────────────────────────

@register("bar")
def _bar(df: pd.DataFrame, cfg: dict) -> go.Figure:
    return px.bar(
        df,
        x=cfg.get("x"),
        y=cfg.get("y"),
        color=cfg.get("group_by"),
        barmode="group",
    )


@register("line")
def _line(df: pd.DataFrame, cfg: dict) -> go.Figure:
    df_sorted = df.sort_values(cfg["x"]) if cfg.get("x") and cfg["x"] in df.columns else df
    return px.line(
        df_sorted,
        x=cfg.get("x"),
        y=cfg.get("y"),
        color=cfg.get("group_by"),
        markers=True,
    )


@register("histogram")
def _histogram(df: pd.DataFrame, cfg: dict) -> go.Figure:
    return px.histogram(
        df,
        x=cfg.get("x") or cfg.get("y"),
        color=cfg.get("group_by"),
        nbins=30,
    )


@register("pie")
def _pie(df: pd.DataFrame, cfg: dict) -> go.Figure:
    return px.pie(
        df,
        names=cfg.get("x"),
        values=cfg.get("y"),
    )


@register("scatter")
def _scatter(df: pd.DataFrame, cfg: dict) -> go.Figure:
    return px.scatter(
        df,
        x=cfg.get("x"),
        y=cfg.get("y"),
        color=cfg.get("group_by"),
        trendline="ols" if cfg.get("group_by") is None else None,
    )
