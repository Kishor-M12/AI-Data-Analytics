"""
chunker.py — Convert a DataFrame into text chunks for RAG indexing.

Strategy:
  - Column summary chunk: describes each column's type and statistics
  - Row chunks: batch rows into text blocks (e.g., 10 rows per chunk)
  - Global summary chunk: overall dataset description

Each chunk is a dict with 'id', 'text', and 'metadata'.
"""
import pandas as pd
import numpy as np
from typing import Iterator
from backend.utils.logger import get_logger

logger = get_logger(__name__)

ROWS_PER_CHUNK = 10  # Rows grouped per text chunk


def chunk_dataframe(df: pd.DataFrame, dataset_name: str = "dataset") -> list[dict]:
    """
    Convert DataFrame into a list of text chunks for embedding.

    Args:
        df: Cleaned DataFrame
        dataset_name: Name label for metadata

    Returns:
        List of chunk dicts: {id, text, metadata}
    """
    chunks = []

    # 1. Global summary chunk
    chunks.append(_global_summary_chunk(df, dataset_name))

    # 2. Column-level summary chunks
    chunks.extend(_column_chunks(df, dataset_name))

    # 3. Row batch chunks
    chunks.extend(_row_chunks(df, dataset_name))

    logger.info(f"Created {len(chunks)} chunks from '{dataset_name}'")
    return chunks


def _global_summary_chunk(df: pd.DataFrame, name: str) -> dict:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    dt_cols = df.select_dtypes(include="datetime").columns.tolist()

    text = (
        f"Dataset '{name}' contains {len(df)} rows and {len(df.columns)} columns. "
        f"Numeric columns: {num_cols}. "
        f"Categorical columns: {cat_cols}. "
        f"Datetime columns: {dt_cols}. "
        f"Column names: {list(df.columns)}."
    )
    return {"id": f"{name}_summary", "text": text, "metadata": {"type": "summary", "dataset": name}}


def _column_chunks(df: pd.DataFrame, name: str) -> list[dict]:
    chunks = []
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)

        if pd.api.types.is_numeric_dtype(series):
            stats = series.describe()
            text = (
                f"Column '{col}' (numeric) in dataset '{name}': "
                f"min={stats['min']:.2f}, max={stats['max']:.2f}, "
                f"mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                f"nulls={series.isnull().sum()}."
            )
        elif pd.api.types.is_datetime64_any_dtype(series):
            text = (
                f"Column '{col}' (datetime) in dataset '{name}': "
                f"range from {series.min()} to {series.max()}, "
                f"nulls={series.isnull().sum()}."
            )
        else:
            top = series.value_counts().head(5).to_dict()
            text = (
                f"Column '{col}' (categorical) in dataset '{name}': "
                f"unique values={series.nunique()}, "
                f"top values={top}, nulls={series.isnull().sum()}."
            )

        chunks.append({
            "id": f"{name}_col_{col}",
            "text": text,
            "metadata": {"type": "column", "column": col, "dtype": dtype, "dataset": name},
        })
    return chunks


def _row_chunks(df: pd.DataFrame, name: str) -> list[dict]:
    """Chunk rows in batches. Each chunk is a mini table as plain text."""
    chunks = []
    df_str = df.astype(str)  # Stringify for uniform text conversion

    for batch_start in range(0, len(df), ROWS_PER_CHUNK):
        batch = df_str.iloc[batch_start: batch_start + ROWS_PER_CHUNK]
        rows_text = batch.to_string(index=False)
        text = f"Data rows {batch_start}–{batch_start + len(batch) - 1} from dataset '{name}':\n{rows_text}"
        chunks.append({
            "id": f"{name}_rows_{batch_start}",
            "text": text,
            "metadata": {
                "type": "rows",
                "row_start": batch_start,
                "row_end": batch_start + len(batch) - 1,
                "dataset": name,
            },
        })
    return chunks
