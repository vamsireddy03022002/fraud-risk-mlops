from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

LABEL_COL = "is_fraud"
TIME_COL = "trans_date_trans_time"


@dataclass(frozen=True)
class FeatureConfig:
    category_top_k: int = 20  # cap categories to avoid exploding feature space
    include_merchant: bool = False  # high cardinality; keep False for Day 3


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found")

    y = df[LABEL_COL].astype(int)
    X_raw = df.drop(columns=[LABEL_COL])
    return X_raw, y


def build_features(
    df: pd.DataFrame,
    config: FeatureConfig = FeatureConfig(),
) -> pd.DataFrame:
    """
    Deterministic feature pipeline.
    - NO label usage.
    - No cross-row aggregates today.
    - Produces stable column order.
    """
    if TIME_COL not in df.columns:
        raise ValueError(f"Expected datetime column '{TIME_COL}' not found")

    X = df.copy()

    # Ensure datetime
    X[TIME_COL] = pd.to_datetime(X[TIME_COL], errors="raise")

    # Datetime features
    X["tx_hour"] = X[TIME_COL].dt.hour.astype(int)
    X["tx_dayofweek"] = X[TIME_COL].dt.dayofweek.astype(int)

    # Drop raw timestamp (not directly model-friendly)
    X = X.drop(columns=[TIME_COL])

    # Numeric columns to keep
    numeric_cols = [
        "amt",
        "city_pop",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
        "unix_time",
        "cc_num",
    ]

    # Categorical columns: start small (safe)
    # merchant is high-cardinality; skip for Day 3
    cat_cols = ["gender", "category"]

    # Safety: ensure expected columns exist
    for col in numeric_cols + cat_cols:
        if col not in X.columns:
            raise ValueError(f"Expected column '{col}' not found in input")

    X_num = X[numeric_cols].copy()

    # One-hot encode gender (tiny)
    X_gender = pd.get_dummies(X["gender"], prefix="gender", dtype=int)

    # One-hot encode category, but cap to top-k
    top_categories = (
        X["category"].value_counts().head(config.category_top_k).index.tolist()
    )
    X_cat = X["category"].where(X["category"].isin(top_categories), other="__OTHER__")
    X_cat_ohe = pd.get_dummies(X_cat, prefix="category", dtype=int)

    # Combine with deterministic column ordering
    X_out = pd.concat([X_num, X_gender, X_cat_ohe], axis=1)

    # Sort columns to ensure stable ordering across runs
    X_out = X_out.reindex(sorted(X_out.columns), axis=1)

    return X_out


def get_feature_columns(
    df: pd.DataFrame,
    config: FeatureConfig = FeatureConfig(),
) -> List[str]:
    X_raw, _ = split_xy(df)
    X_feat = build_features(X_raw, config=config)
    return list(X_feat.columns)
