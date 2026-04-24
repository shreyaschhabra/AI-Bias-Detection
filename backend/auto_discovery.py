"""
auto_discovery.py
-----------------
Detects sensitive / protected attributes in a DataFrame automatically using
two complementary strategies:

  1. Semantic Scanner  – matches column names against a curated keyword list.
  2. Statistical Profiler – finds low-cardinality columns with high mutual
     information to the target (fallback when semantic scan returns nothing).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# ---------------------------------------------------------------------------
# Keyword list – extend as needed
# ---------------------------------------------------------------------------
_SENSITIVE_KEYWORDS: list[str] = [
    "race", "racial", "ethnicity", "ethnic",
    "gender", "sex", "sexuality", "orientation",
    "age", "dob", "birthdate", "birth_date",
    "religion", "religious", "faith",
    "nationality", "national", "citizenship",
    "marital", "marriage", "married",
    "disability", "disabled", "handicap",
    "zip", "zipcode", "postcode", "postal",
    "income", "salary",              # common proxies
    "weight", "height",              # physical proxies
]


def _tokenise(col: str) -> list[str]:
    """Split a column name into lowercase tokens on non-alphanumeric chars."""
    return re.split(r"[^a-z0-9]+", col.strip().lower())


def _semantic_scan(df: pd.DataFrame, target_col: str) -> list[str]:
    """
    Return all column names (excluding target) whose tokens overlap with
    any keyword in _SENSITIVE_KEYWORDS, or whose full name contains a keyword
    as a substring.
    """
    hits: list[str] = []
    for col in df.columns:
        if col == target_col:
            continue
        tokens = _tokenise(col)
        col_lower = col.lower()
        for kw in _SENSITIVE_KEYWORDS:
            # exact token match OR substring (e.g. "zipcode" contains "zip")
            if kw in tokens or kw in col_lower:
                hits.append(col)
                break
    return hits


def _statistical_profile(
    df: pd.DataFrame,
    target_col: str,
    max_cardinality: int = 10,
    top_k: int = 3,
) -> list[str]:
    """
    Fallback: rank non-target columns by mutual information with target.
    Return the top-k low-cardinality (nunique <= max_cardinality) columns.
    """
    candidate_cols = [
        c for c in df.columns
        if c != target_col and df[c].nunique() <= max_cardinality
    ]
    if not candidate_cols:
        return []

    # Encode target as int (handles string labels)
    target_series = df[target_col]
    try:
        y = target_series.astype(int).to_numpy()
    except (ValueError, TypeError):
        y = pd.Categorical(target_series).codes

    X_candidates = pd.DataFrame(index=df.index)
    for c in candidate_cols:
        try:
            X_candidates[c] = pd.Categorical(df[c]).codes
        except Exception:
            X_candidates[c] = df[c].fillna(0).astype(float)

    try:
        mi_scores = mutual_info_classif(
            X_candidates.to_numpy(), y, random_state=42
        )
    except Exception:
        return []

    # Filter to columns above the median MI score
    median_mi = float(np.median(mi_scores))
    ranked = sorted(
        zip(candidate_cols, mi_scores),
        key=lambda x: x[1],
        reverse=True,
    )
    result = [col for col, score in ranked if score > median_mi]
    return result[:top_k]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_sensitive_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """
    Auto-detect which columns are likely protected/sensitive attributes.

    Strategy:
      1. Semantic scan of column headers.
      2. If nothing found, fall back to statistical profiler.

    Returns a list of column names (may be empty if truly nothing detected).
    """
    hits = _semantic_scan(df, target_col)
    if hits:
        return hits

    # Fallback: statistical profiler
    return _statistical_profile(df, target_col)
