import io
import json
import logging
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from auto_discovery import detect_sensitive_columns
from counterfactual import compute_counterfactual_flip_rate
from fairness_metrics import (
    compute_group_metrics,
    compute_individual_metrics,
    compute_intersectional_metrics,
)
from model_wrapper import ModelWrapper
from root_cause import compute_root_cause
from schemas import (
    AnalyzeResponse,
    FeatureInfo,
    FairnessMetrics,
    IntersectionalMetric,
    MitigateResponse,
    SummaryRequest,
    SummaryResponse,
)
from mitigation import apply_reweighing_and_retrain
from suggestions import generate_suggestions

logger = logging.getLogger("autobias")

app = FastAPI(title="AutoBias API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# SHAP helper
# ---------------------------------------------------------------------------

def _compute_shap_values(
    wrapper: ModelWrapper,
    X: pd.DataFrame,
    max_rows: int = 2000,
    kernel_background: int = 100,
) -> tuple[list[dict], bool]:
    """
    Compute per-row SHAP values for the positive class (index 1).

    Strategy order (fastest → slowest):
      1. TreeExplainer   – tree-based models (RF, GB, XGB, LGBM …)
      2. LinearExplainer – linear models (LogisticRegression, SVC with linear kernel …)
      3. KernelExplainer – universal fallback (slow; sampled)

    Returns:
        (shap_dicts, truncated)
        shap_dicts  – list of {feature: float} dicts, one per row in X
        truncated   – True when the dataset was down-sampled for performance
    """
    try:
        import shap  # local import so server starts without shap installed
    except ImportError:
        logger.warning("shap not installed – returning empty SHAP values")
        return [], False

    feature_names: list[str] = list(X.columns)
    X_np = X.to_numpy()

    truncated = len(X) > max_rows
    if truncated:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        X_sample = X.iloc[idx].reset_index(drop=True)
        X_np_sample = X_sample.to_numpy()
    else:
        X_sample = X.copy()
        X_np_sample = X_np

    raw_model = wrapper.model
    shap_vals = None

    # 1. TreeExplainer
    try:
        explainer = shap.TreeExplainer(raw_model)
        sv = explainer.shap_values(X_np_sample)
        # sv may be list-of-arrays (one per class) or a single array
        if isinstance(sv, list):
            shap_vals = sv[1]   # positive class
        else:
            shap_vals = sv
        logger.info("SHAP: used TreeExplainer")
    except Exception:
        pass

    # 2. LinearExplainer
    if shap_vals is None:
        try:
            explainer = shap.LinearExplainer(raw_model, X_np_sample)
            sv = explainer.shap_values(X_np_sample)
            if isinstance(sv, list):
                shap_vals = sv[1]
            else:
                shap_vals = sv
            logger.info("SHAP: used LinearExplainer")
        except Exception:
            pass

    # 3. KernelExplainer (universal fallback)
    if shap_vals is None:
        try:
            # Use predict_proba if available, else predict
            if hasattr(raw_model, "predict_proba"):
                predict_fn = lambda x: raw_model.predict_proba(x)[:, 1]
            else:
                predict_fn = lambda x: raw_model.predict(x).astype(float)

            bg_size = min(kernel_background, len(X_np_sample))
            background = shap.kmeans(X_np_sample, bg_size)
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_vals = explainer.shap_values(X_np_sample, silent=True)
            logger.info("SHAP: used KernelExplainer (background=%d)", bg_size)
        except Exception as exc:
            logger.error("SHAP KernelExplainer failed: %s", exc)
            return [], truncated

    # Build list of per-row dicts
    if shap_vals is None or len(shap_vals) == 0:
        return [], truncated

    # Ensure shap_vals is exactly 2D: (n_samples, n_features)
    shap_vals = np.array(shap_vals)
    if len(shap_vals.shape) == 3:
        # typical binary classification tree output array: (samples, features, classes)
        # we take class index 1
        shap_vals = shap_vals[:, :, -1]

    result: list[dict] = []
    for row in shap_vals:
        result.append({f: round(float(v), 6) for f, v in zip(feature_names, row)})

    return result, truncated


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/status")
async def status():
    return {"status": "ok", "version": "0.2.0"}


@app.post("/api/analyze", response_model=AnalyzeResponse)
@app.post("/analyze", response_model=AnalyzeResponse)   # keep old path for compatibility
async def analyze(
    file: UploadFile = File(..., description="Dataset CSV"),
    model: UploadFile = File(..., description="Pickled sklearn model"),
    target_column: str = Form(...),
    sensitive_column: Optional[str] = Form(None),       # NOW OPTIONAL
    privileged_value: Optional[str] = Form(None),
    positive_label: str = Form("1"),
):
    # ------------------------------------------------------------------ #
    # 1. Load CSV
    # ------------------------------------------------------------------ #
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    if target_column not in df.columns:
        raise HTTPException(
            400,
            f"target_column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}",
        )

    # ------------------------------------------------------------------ #
    # 2. Auto-discover sensitive columns if not provided
    # ------------------------------------------------------------------ #
    if sensitive_column:
        # User supplied it — validate then wrap in list
        if sensitive_column not in df.columns:
            raise HTTPException(
                400,
                f"Column '{sensitive_column}' not found in CSV. "
                f"Available columns: {list(df.columns)}",
            )
        detected_sensitive_cols: list[str] = [sensitive_column]
    else:
        detected_sensitive_cols = detect_sensitive_columns(df, target_column)
        if not detected_sensitive_cols:
            raise HTTPException(
                422,
                "Could not auto-detect any sensitive attribute. "
                "Please provide 'sensitive_column' explicitly.",
            )
        logger.info("Auto-detected sensitive columns: %s", detected_sensitive_cols)

    # Use the first detected column as the primary sensitive column
    primary_sensitive_col = detected_sensitive_cols[0]

    # ------------------------------------------------------------------ #
    # 3. Determine privileged_value for the primary column
    # ------------------------------------------------------------------ #
    if privileged_value is None:
        # Heuristic: most-frequent value in the primary sensitive column
        privileged_value = str(df[primary_sensitive_col].value_counts().idxmax())
        logger.info(
            "Auto-selected privileged_value='%s' for column '%s'",
            privileged_value,
            primary_sensitive_col,
        )

    # ------------------------------------------------------------------ #
    # 4. Load model
    # ------------------------------------------------------------------ #
    try:
        raw_model = joblib.load(io.BytesIO(await model.read()))
    except Exception as e:
        raise HTTPException(400, f"Could not load model file: {e}")

    # ------------------------------------------------------------------ #
    # 5. Prepare arrays
    # ------------------------------------------------------------------ #
    feature_cols = [c for c in df.columns if c != target_column]
    if not feature_cols:
        raise HTTPException(400, "No feature columns found after removing the target.")

    X = df[feature_cols].copy()
    y_true = df[target_column].to_numpy()

    try:
        wrapper = ModelWrapper(raw_model, feature_cols)
    except ValueError as e:
        raise HTTPException(400, str(e))

    y_pred = wrapper.predict(X)

    # Validate binary classification
    unique_preds = np.unique(y_pred)
    if len(unique_preds) > 2:
        raise HTTPException(
            400,
            f"Model produces {len(unique_preds)} unique output values. "
            "Only binary classification models are supported.",
        )

    try:
        pos = int(positive_label)
    except (ValueError, TypeError):
        pos = positive_label

    y_pred_int = y_pred.astype(int)
    try:
        y_true_int = y_true.astype(int)
    except (ValueError, TypeError):
        y_true_int = (y_true.astype(str) == str(positive_label)).astype(int)
        y_pred_int = (y_pred.astype(str) == str(positive_label)).astype(int)
        pos = 1

    # ------------------------------------------------------------------ #
    # 6. Group fairness metrics (SPD, DI – AIF360, primary sensitive col)
    # ------------------------------------------------------------------ #
    df_for_metrics = df[feature_cols + [target_column]].copy()
    group_result = compute_group_metrics(
        df=df_for_metrics,
        y_pred=y_pred_int,
        target_col=target_column,
        sensitive_col=primary_sensitive_col,
        privileged_value=privileged_value,
        positive_label=pos,
    )

    # ------------------------------------------------------------------ #
    # 7. Individual/distributional metrics (EOD, consistency, GEE)
    # ------------------------------------------------------------------ #
    X_no_sensitive = df[[c for c in feature_cols if c != primary_sensitive_col]].copy()
    prot_series = df[primary_sensitive_col]
    individual_result = compute_individual_metrics(
        X=X_no_sensitive,
        y_true=y_true_int,
        y_pred=y_pred_int,
        prot_attr_series=prot_series,
        priv_label=privileged_value,
    )

    # ------------------------------------------------------------------ #
    # 8. Intersectional fairness metrics (NEW)
    # ------------------------------------------------------------------ #
    intersectional_raw = compute_intersectional_metrics(
        df=df[feature_cols].copy(),
        y_pred=y_pred_int,
        sensitive_cols=detected_sensitive_cols,
        target_col=target_column,
        positive_label=pos,
    )

    # ------------------------------------------------------------------ #
    # 9. Counterfactual flip rate
    # ------------------------------------------------------------------ #
    flip_rate = compute_counterfactual_flip_rate(
        wrapper=wrapper,
        X=X,
        sensitive_col=primary_sensitive_col,
        original_preds=y_pred_int,
    )

    # ------------------------------------------------------------------ #
    # 10. Root-cause analysis
    # ------------------------------------------------------------------ #
    top_features_raw = compute_root_cause(
        wrapper=wrapper,
        X=X,
        y_true=y_true_int,
        sensitive_col=primary_sensitive_col,
    )

    # ------------------------------------------------------------------ #
    # 11. SHAP values (local explainability – NEW)
    # ------------------------------------------------------------------ #
    shap_dicts, shap_truncated = _compute_shap_values(wrapper, X)

    # ------------------------------------------------------------------ #
    # 12. Suggestions
    # ------------------------------------------------------------------ #
    all_metrics = {
        "statistical_parity_difference": group_result["statistical_parity_difference"],
        "disparate_impact": group_result["disparate_impact"],
        "equal_opportunity_difference": individual_result["equal_opportunity_difference"],
    }
    suggestions = generate_suggestions(all_metrics, flip_rate, top_features_raw)

    # ------------------------------------------------------------------ #
    # 13. Build response
    # ------------------------------------------------------------------ #
    return AnalyzeResponse(
        metrics=FairnessMetrics(
            statistical_parity_difference=group_result["statistical_parity_difference"],
            disparate_impact=group_result["disparate_impact"],
            equal_opportunity_difference=individual_result["equal_opportunity_difference"],
            consistency_score=individual_result["consistency_score"],
            generalized_entropy_error=individual_result["generalized_entropy_error"],
        ),
        counterfactual_flip_rate=flip_rate,
        top_features=[FeatureInfo(**f) for f in top_features_raw],
        suggestions=suggestions,
        group_selection_rates=group_result["group_selection_rates"],
        # Phase 1 additions
        detected_sensitive_cols=detected_sensitive_cols,
        intersectional_metrics=[IntersectionalMetric(**m) for m in intersectional_raw],
        shap_values=shap_dicts,
        shap_truncated=shap_truncated,
    )


@app.post("/api/mitigate", response_model=MitigateResponse)
async def mitigate(
    file: UploadFile = File(..., description="Dataset CSV"),
    model: UploadFile = File(..., description="Pickled sklearn model"),
    target_column: str = Form(...),
    sensitive_column: str = Form(...),
    privileged_value: str = Form(...),
    positive_label: str = Form("1"),
):
    # ------------------------------------------------------------------ #
    # 1. Load CSV and Model
    # ------------------------------------------------------------------ #
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    try:
        raw_model = joblib.load(io.BytesIO(await model.read()))
    except Exception as e:
        raise HTTPException(400, f"Could not load model file: {e}")

    # ------------------------------------------------------------------ #
    # 2. Extract features and evaluate unmitigated predictions
    # ------------------------------------------------------------------ #
    feature_cols = [c for c in df.columns if c != target_column]
    if not feature_cols:
        raise HTTPException(400, "No feature columns found.")

    X = df[feature_cols].copy()
    y_true = df[target_column].to_numpy()

    try:
        wrapper_before = ModelWrapper(raw_model, feature_cols)
    except ValueError as e:
        raise HTTPException(400, str(e))
        
    y_pred_before = wrapper_before.predict(X)

    unique_preds = np.unique(y_pred_before)
    if len(unique_preds) > 2:
        raise HTTPException(400, "Only binary classification models are supported.")

    try:
        pos = int(positive_label)
    except (ValueError, TypeError):
        pos = positive_label

    y_pred_int_before = y_pred_before.astype(int)
    try:
        y_true_int = y_true.astype(int)
    except (ValueError, TypeError):
        y_true_int = (y_true.astype(str) == str(positive_label)).astype(int)
        y_pred_int_before = (y_pred_before.astype(str) == str(positive_label)).astype(int)
        pos = 1

    # Before metrics
    df_for_metrics = df[feature_cols + [target_column]].copy()
    group_result_before = compute_group_metrics(
        df=df_for_metrics,
        y_pred=y_pred_int_before,
        target_col=target_column,
        sensitive_col=sensitive_column,
        privileged_value=privileged_value,
        positive_label=pos,
    )
    
    X_no_sensitive = df[[c for c in feature_cols if c != sensitive_column]].copy()
    prot_series = df[sensitive_column]
    indiv_result_before = compute_individual_metrics(
        X=X_no_sensitive,
        y_true=y_true_int,
        y_pred=y_pred_int_before,
        prot_attr_series=prot_series,
        priv_label=privileged_value,
    )

    # ------------------------------------------------------------------ #
    # 3. Apply Mitigation (Reweighing) and Retrain
    # ------------------------------------------------------------------ #
    try:
        new_model, model_b64 = apply_reweighing_and_retrain(
            df=df,
            target_col=target_column,
            sensitive_col=sensitive_column,
            privileged_value=privileged_value,
            positive_label=positive_label,
            raw_model=raw_model,
            feature_cols=feature_cols,
        )
    except ValueError as ve:
        raise HTTPException(400, str(ve))
    except Exception as e:
        logger.error("Mitigation error: %s", e)
        raise HTTPException(500, "Error during mitigation retraining.")

    # ------------------------------------------------------------------ #
    # 4. Evaluate mitigated predictions
    # ------------------------------------------------------------------ #
    wrapper_after = ModelWrapper(new_model, feature_cols)
    y_pred_after = wrapper_after.predict(X)

    y_pred_int_after = y_pred_after.astype(int)
    if len(np.unique(y_pred_int_after)) <= 2:
        try:
            # Re-apply the same parsing as above
            if not isinstance(y_true[0], (int, np.integer, float, np.floating)):
                y_pred_int_after = (y_pred_after.astype(str) == str(positive_label)).astype(int)
        except Exception:
            pass

    group_result_after = compute_group_metrics(
        df=df_for_metrics,
        y_pred=y_pred_int_after,
        target_col=target_column,
        sensitive_col=sensitive_column,
        privileged_value=privileged_value,
        positive_label=pos,
    )
    
    indiv_result_after = compute_individual_metrics(
        X=X_no_sensitive,
        y_true=y_true_int,
        y_pred=y_pred_int_after,
        prot_attr_series=prot_series,
        priv_label=privileged_value,
    )

    return MitigateResponse(
        before_metrics=FairnessMetrics(
            statistical_parity_difference=group_result_before["statistical_parity_difference"],
            disparate_impact=group_result_before["disparate_impact"],
            equal_opportunity_difference=indiv_result_before["equal_opportunity_difference"],
            consistency_score=indiv_result_before["consistency_score"],
            generalized_entropy_error=indiv_result_before["generalized_entropy_error"],
        ),
        after_metrics=FairnessMetrics(
            statistical_parity_difference=group_result_after["statistical_parity_difference"],
            disparate_impact=group_result_after["disparate_impact"],
            equal_opportunity_difference=indiv_result_after["equal_opportunity_difference"],
            consistency_score=indiv_result_after["consistency_score"],
            generalized_entropy_error=indiv_result_after["generalized_entropy_error"],
        ),
        mitigated_model_base64=model_b64,
    )


# ---------------------------------------------------------------------------
# Phase 3: Automated Data Storytelling Endpoint
# ---------------------------------------------------------------------------

@app.post("/api/generate-summary", response_model=SummaryResponse)
async def generate_summary(req: SummaryRequest):
    try:
        from llm_client import generate_executive_summary
        
        # Convert Pydantic to pure dicts for json.dumps
        metrics_dict = req.metrics.model_dump()
        features_list = [f.model_dump() for f in req.top_features]

        summary_text = generate_executive_summary(
            metrics=metrics_dict,
            top_features=features_list
        )
        return SummaryResponse(summary=summary_text)
    except Exception as e:
        logger.error("LLM Generation error: %s", e)
        raise HTTPException(500, f"Error generating summary: {e}")


# ---------------------------------------------------------------------------
# Phase 4: Live Counterfactual Prediction
# ---------------------------------------------------------------------------

@app.post("/api/predict-counterfactual")
async def predict_counterfactual(
    model: UploadFile = File(..., description="Pickled sklearn model"),
    row_data: str = Form(..., description="JSON string of the single row"),
    feature_cols: str = Form(..., description="JSON string of feature columns array"),
):
    try:
        raw_model = joblib.load(io.BytesIO(await model.read()))
    except Exception as e:
        raise HTTPException(400, f"Could not load model file: {e}")

    try:
        row_dict = json.loads(row_data)
        features_list = json.loads(feature_cols)
    except Exception as e:
        raise HTTPException(400, f"Could not parse JSON forms: {e}")

    df_row = pd.DataFrame([row_dict])
    
    # Handle missing features just in case
    for f in features_list:
        if f not in df_row.columns:
            df_row[f] = 0

    X = df_row[features_list].copy()

    try:
        wrapper = ModelWrapper(raw_model, features_list)
        pred = wrapper.predict(X)[0]
    except Exception as e:
        raise HTTPException(400, f"Prediction error: {e}")

    try:
        pred_val = int(pred)
    except (ValueError, TypeError):
        pred_val = str(pred)

    prob_val = None
    if hasattr(raw_model, "predict_proba"):
        try:
            probs = raw_model.predict_proba(X)
            # Try to grab the positive class probability
            # Assuming typical binary classification shape (1, 2)
            if probs.shape == (1, 2):
                prob_val = float(probs[0][1])
            else:
                prob_val = float(probs[0][-1])
        except Exception:
            pass

    return {
        "prediction": pred_val,
        "probability": prob_val
    }



