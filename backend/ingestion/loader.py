"""
loader.py — Load datasets from files (CSV/Excel/JSON) or database connections.
Returns a standardized pandas DataFrame.
"""
import io
import pandas as pd
from pathlib import Path
from typing import Optional
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def load_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Parse uploaded file bytes into a DataFrame.

    Supports: .csv, .xlsx, .xls, .json
    Args:
        file_bytes: Raw bytes of the uploaded file
        filename: Original filename (used to detect extension)
    Returns:
        pd.DataFrame
    Raises:
        ValueError: If file type is unsupported
    """
    ext = Path(filename).suffix.lower()
    buffer = io.BytesIO(file_bytes)

    if ext == ".csv":
        df = pd.read_csv(buffer, encoding_errors="replace")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(buffer)
    elif ext == ".json":
        df = pd.read_json(buffer)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use CSV, Excel, or JSON.")

    logger.info(f"Loaded '{filename}': {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_from_database(connection_string: str, query: str) -> pd.DataFrame:
    """
    Load data from a SQL database using SQLAlchemy.

    Args:
        connection_string: SQLAlchemy-compatible connection URI
        query: SQL query to execute
    Returns:
        pd.DataFrame
    """
    from sqlalchemy import create_engine
    engine = create_engine(connection_string)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    logger.info(f"Loaded {len(df)} rows from database via query.")
    return df
