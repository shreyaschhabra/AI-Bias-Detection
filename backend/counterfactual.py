import numpy as np
import pandas as pd

from model_wrapper import ModelWrapper


def compute_counterfactual_flip_rate(
    wrapper: ModelWrapper,
    X: pd.DataFrame,
    sensitive_col: str,
    original_preds: np.ndarray,
    sample_size: int = 1000,
) -> float:
    """
    For each row (up to sample_size), flip the sensitive attribute to every
    other possible value and re-run model.predict on the full feature row
    (sensitive column is part of the model's feature set).
    Returns the fraction of rows where at least one flip changes the prediction.
    """
    sensitive_values = list(X[sensitive_col].unique())
    if len(sensitive_values) < 2:
        return 0.0

    # Sample if needed — keep a positional index into original_preds
    if len(X) > sample_size:
        idx = X.sample(n=sample_size, random_state=42).index
        X_sample = X.loc[idx].reset_index(drop=True)
        orig_preds_sample = original_preds[idx].copy()
    else:
        X_sample = X.reset_index(drop=True)
        orig_preds_sample = np.array(original_preds)

    # Vectorized path for binary sensitive attribute
    if len(sensitive_values) == 2:
        val_a, val_b = sensitive_values
        X_flipped = X_sample.copy()
        X_flipped[sensitive_col] = X_sample[sensitive_col].map(
            {val_a: val_b, val_b: val_a}
        )
        flipped_preds = wrapper.predict(X_flipped)
        changed = int((flipped_preds != orig_preds_sample).sum())
        return changed / len(X_sample)

    # Multi-category: row-by-row loop
    changed = 0
    for i, row in X_sample.iterrows():
        orig_val = row[sensitive_col]
        orig_pred = orig_preds_sample[i]
        for alt_val in sensitive_values:
            if alt_val == orig_val:
                continue
            cf_row = row.copy()
            cf_row[sensitive_col] = alt_val
            cf_df = pd.DataFrame([cf_row], columns=X_sample.columns)
            cf_pred = wrapper.predict(cf_df)[0]
            if cf_pred != orig_pred:
                changed += 1
                break
    return changed / len(X_sample)
