import itertools

import numpy as np
import pandas as pd


def _encode_sensitive(
    df: pd.DataFrame, sensitive_col: str, privileged_value
) -> tuple[int, pd.DataFrame]:
    """
    Map the sensitive column to 0/1 integers.
    privileged_value → 1, all others → 0.
    Returns (encoded_privileged_value, modified_df).
    """
    df_enc = df.copy()
    df_enc[sensitive_col] = (df_enc[sensitive_col].astype(str) == str(privileged_value)).astype(int)
    return 1, df_enc


def compute_group_metrics(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    target_col: str,
    sensitive_col: str,
    privileged_value,
    positive_label,
) -> dict:
    """
    Compute SPD, DI, and group selection rates using AIF360 BinaryLabelDatasetMetric.
    Returns dict with keys: statistical_parity_difference, disparate_impact,
    group_selection_rates.
    """
    from aif360.datasets import StandardDataset
    from aif360.metrics import BinaryLabelDatasetMetric

    # Build a copy of the df with predictions as the label column
    df_pred = df.copy()
    df_pred[target_col] = y_pred

    priv_encoded, df_enc = _encode_sensitive(df_pred, sensitive_col, privileged_value)
    unpriv_encoded = 0

    # Ensure positive_label is the right type
    try:
        pos = int(positive_label)
    except (ValueError, TypeError):
        pos = positive_label

    dataset = StandardDataset(
        df_enc,
        label_name=target_col,
        favorable_classes=[pos],
        protected_attribute_names=[sensitive_col],
        privileged_classes=[[priv_encoded]],
    )

    priv_groups = [{sensitive_col: priv_encoded}]
    unpriv_groups = [{sensitive_col: unpriv_encoded}]

    metric = BinaryLabelDatasetMetric(
        dataset,
        unprivileged_groups=unpriv_groups,
        privileged_groups=priv_groups,
    )

    spd = float(metric.statistical_parity_difference())
    di = float(metric.disparate_impact())

    # Compute selection rates per group for bar chart
    priv_mask = df_enc[sensitive_col] == priv_encoded
    unpriv_mask = df_enc[sensitive_col] == unpriv_encoded
    priv_rate = float(y_pred[priv_mask].mean()) if priv_mask.any() else 0.0
    unpriv_rate = float(y_pred[unpriv_mask].mean()) if unpriv_mask.any() else 0.0

    return {
        "statistical_parity_difference": spd,
        "disparate_impact": di,
        "group_selection_rates": {
            "privileged": priv_rate,
            "unprivileged": unpriv_rate,
        },
    }


def compute_individual_metrics(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prot_attr_series: pd.Series,
    priv_label,
) -> dict:
    """
    Compute EOD, consistency_score, and generalized_entropy_error.
    Uses AIF360 sklearn-style API.
    """
    from aif360.sklearn.metrics import (
        equal_opportunity_difference,
        consistency_score,
        generalized_entropy_error,
    )

    # Encode protected attribute to 0/1
    prot_encoded = (prot_attr_series.astype(str) == str(priv_label)).astype(int)

    y_true_int = y_true.astype(int)
    y_pred_int = y_pred.astype(int)

    # EOD: requires y_true and y_pred with prot_attr
    try:
        eod = float(
            equal_opportunity_difference(
                y_true_int,
                y_pred_int,
                prot_attr=prot_encoded,
                priv_group=1,
            )
        )
    except Exception:
        eod = 0.0

    # Consistency score: pass numpy array for X
    try:
        cons = float(consistency_score(X.to_numpy(), y_pred_int))
    except Exception:
        cons = 1.0

    # Generalized entropy error
    try:
        gee = float(generalized_entropy_error(y_true_int, y_pred_int))
    except Exception:
        gee = 0.0

    return {
        "equal_opportunity_difference": eod,
        "consistency_score": cons,
        "generalized_entropy_error": gee,
    }


def compute_intersectional_metrics(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    sensitive_cols: list[str],
    target_col: str,
    positive_label,
    max_subgroups: int = 50,
    min_subgroup_size: int = 5,
) -> list[dict]:
    """
    Compute DI and SPD for every unique combination of values across
    `sensitive_cols`.  Each subgroup is compared to the *complement* (all
    rows NOT in that subgroup), so the metric is always well-defined.

    Returns a list of dicts sorted by DI ascending (most biased first).
    Each dict has:
        subgroup         – human-readable label, e.g. "race=Black & gender=Female"
        selection_rate   – positive-prediction rate for this subgroup
        complement_rate  – positive-prediction rate for everyone else
        disparate_impact – subgroup_rate / complement_rate  (NaN if 0-denominator)
        statistical_parity_difference – subgroup_rate - complement_rate
        size             – number of rows in the subgroup
    """
    # Clamp to at most 3 columns to avoid combinatorial explosion
    cols = sensitive_cols[:3]

    # Work on a copy with predictions
    df_work = df.copy()
    df_work["__pred__"] = y_pred

    # Positive-label normalisation
    try:
        pos = int(positive_label)
    except (ValueError, TypeError):
        pos = positive_label

    def sel_rate(mask: pd.Series) -> float:
        sub = df_work.loc[mask, "__pred__"]
        if len(sub) == 0:
            return 0.0
        return float((sub == pos).sum() / len(sub))

    # Build the unique-value lists per column
    value_lists = [df_work[c].dropna().unique().tolist() for c in cols]

    results: list[dict] = []
    for combo in itertools.product(*value_lists):
        label_parts = [f"{c}={v}" for c, v in zip(cols, combo)]
        label = " & ".join(label_parts)

        # Build mask for this subgroup
        mask = pd.Series(True, index=df_work.index)
        for c, v in zip(cols, combo):
            mask = mask & (df_work[c].astype(str) == str(v))

        size = int(mask.sum())
        if size < min_subgroup_size:
            continue

        complement_mask = ~mask
        sub_rate = sel_rate(mask)
        comp_rate = sel_rate(complement_mask)

        di = (sub_rate / comp_rate) if comp_rate > 0 else float("nan")
        spd = sub_rate - comp_rate

        results.append(
            {
                "subgroup": label,
                "selection_rate": round(sub_rate, 4),
                "complement_rate": round(comp_rate, 4),
                "disparate_impact": round(di, 4) if not np.isnan(di) else None,
                "statistical_parity_difference": round(spd, 4),
                "size": size,
            }
        )

        if len(results) >= max_subgroups:
            break

    # Sort by DI ascending (most biased first); NaN pushed to end
    results.sort(key=lambda r: (r["disparate_impact"] is None, r["disparate_impact"] or 1.0))
    return results
