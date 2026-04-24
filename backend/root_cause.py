import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance

from model_wrapper import ModelWrapper


def compute_root_cause(
    wrapper: ModelWrapper,
    X: pd.DataFrame,
    y_true: np.ndarray,
    sensitive_col: str,
    top_k: int = 10,
) -> list[dict]:
    """
    Return top_k features ranked by model importance, each annotated with
    normalized mutual information against the sensitive column.
    """
    feature_names = [c for c in X.columns if c != sensitive_col]
    X_features = X[feature_names].copy()

    # Step 1: model-native importances
    importance_map = wrapper.feature_importances(feature_names)

    # Step 2: fallback to permutation importance
    if not importance_map:
        result = permutation_importance(
            wrapper.model,
            X_features.to_numpy(),
            y_true,
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )
        raw = result.importances_mean
        # Clip negatives (random noise) to 0 and normalize
        raw = np.clip(raw, 0, None)
        total = raw.sum() or 1.0
        importance_map = dict(zip(feature_names, raw / total))

    # Step 3: mutual information between each feature and sensitive column
    prot_numeric = pd.Categorical(X[sensitive_col]).codes
    X_filled = X_features.fillna(0).to_numpy()
    mi_scores = mutual_info_classif(X_filled, prot_numeric, random_state=42)

    max_mi = mi_scores.max() if mi_scores.max() > 0 else 1.0
    mi_norm = dict(zip(feature_names, mi_scores / max_mi))

    # Step 4: rank by importance
    ranked = sorted(importance_map.items(), key=lambda x: x[1], reverse=True)
    return [
        {
            "feature": feat,
            "importance": round(float(imp), 4),
            "corr_with_sensitive": round(float(mi_norm.get(feat, 0.0)), 4),
        }
        for feat, imp in ranked[:top_k]
    ]
