"""
cleaner.py — Data quality checks and preprocessing pipeline.
Handles missing values, duplicates, type inference, and outlier flagging.
"""
import pandas as pd
import numpy as np
from typing import Tuple
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Run the full cleaning pipeline on a DataFrame.

    Steps:
        1. Drop fully empty rows/columns
        2. Remove duplicate rows
        3. Infer & fix dtypes (numeric, datetime)
        4. Fill or flag missing values

    Returns:
        Tuple of (cleaned_df, quality_report)
    """
    report = {}
    original_shape = df.shape

    # 1. Drop fully empty rows and columns
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # 2. Remove duplicates
    before_dedup = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = before_dedup - len(df)

    # 3. Infer types
    df = _infer_types(df)

    # 4. Missing values report
    missing = df.isnull().sum()
    report["missing_values"] = missing[missing > 0].to_dict()
    report["missing_pct"] = {
        col: round(cnt / len(df) * 100, 2)
        for col, cnt in report["missing_values"].items()
    }

    # Fill numeric NaNs with column median
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill categorical NaNs with "Unknown"
    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    report["original_shape"] = original_shape
    report["final_shape"] = df.shape
    report["column_types"] = df.dtypes.astype(str).to_dict()

    logger.info(f"Cleaning complete. Shape: {original_shape} → {df.shape}. Report: {report}")
    return df, report


def _infer_types(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to coerce object columns to numeric or datetime."""
    for col in df.select_dtypes(include="object").columns:
        # Try numeric
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() / max(len(df), 1) > 0.8:
            df[col] = converted
            continue
        # Try datetime
        try:
            converted_dt = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            if converted_dt.notna().sum() / max(len(df), 1) > 0.7:
                df[col] = converted_dt
        except Exception:
            pass
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Return a statistical summary dict for the LLM context.
    Includes shape, dtypes, describe stats, top value counts for categoricals.
    """
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_stats": df.describe(include=np.number).round(2).to_dict(),
    }
    cat_cols = df.select_dtypes(include="object").columns
    summary["categorical_top"] = {
        col: df[col].value_counts().head(5).to_dict() for col in cat_cols
    }
    return summary
