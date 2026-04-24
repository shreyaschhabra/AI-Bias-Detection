import base64
import inspect
import io

import joblib
import pandas as pd
import sklearn.base
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import StandardDataset


def apply_reweighing_and_retrain(
    df: pd.DataFrame,
    target_col: str,
    sensitive_col: str,
    privileged_value,
    positive_label,
    raw_model,
    feature_cols: list[str],
) -> tuple[sklearn.base.BaseEstimator, str]:
    """
    Applies AIF360 Reweighing algorithm and retrains the given sklearn model.
    Returns:
        (new_model, model_base64_string)
    """
    df_enc = df.copy()
    # Map sensitive column: privileged_value -> 1, others -> 0
    df_enc[sensitive_col] = (df_enc[sensitive_col].astype(str) == str(privileged_value)).astype(int)

    try:
        pos = int(positive_label)
    except (ValueError, TypeError):
        pos = positive_label

    # Ensure target is encoded nicely. standard dataset handles it but helps to be explicit.
    # We'll just pass the labels in and let favorable_classes do its thing.
    dataset = StandardDataset(
        df_enc,
        label_name=target_col,
        favorable_classes=[pos],
        protected_attribute_names=[sensitive_col],
        privileged_classes=[[1]],
    )

    privileged_groups = [{sensitive_col: 1}]
    unprivileged_groups = [{sensitive_col: 0}]

    rw = Reweighing(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )
    dataset_transf = rw.fit_transform(dataset)

    # Extract new instances weights provided by AIF360
    sample_weights = dataset_transf.instance_weights

    # Retrain
    new_model = sklearn.base.clone(raw_model)

    # Get X and y in original encoding matching what the model expects
    X_train = df[feature_cols].copy()
    y_train = df[target_col].to_numpy()

    try:
        new_model.fit(X_train, y_train, sample_weight=sample_weights)
    except TypeError as e:
        if "sample_weight" in str(e):
            raise ValueError(
                f"Model type {type(new_model).__name__} does not support "
                "sample weights required for Reweighing."
            )
        raise e

    # Serialize to base64 for frontend consumption
    buffer = io.BytesIO()
    joblib.dump(new_model, buffer)
    buffer.seek(0)
    model_b64 = base64.b64encode(buffer.read()).decode("utf-8")

    return new_model, model_b64
