from typing import Optional

from pydantic import BaseModel


class FairnessMetrics(BaseModel):
    statistical_parity_difference: float
    disparate_impact: float
    equal_opportunity_difference: float
    consistency_score: float
    generalized_entropy_error: float


class FeatureInfo(BaseModel):
    feature: str
    importance: float
    corr_with_sensitive: float


class IntersectionalMetric(BaseModel):
    subgroup: str
    selection_rate: float
    complement_rate: float
    disparate_impact: Optional[float]
    statistical_parity_difference: float
    size: int


class AnalyzeResponse(BaseModel):
    metrics: FairnessMetrics
    counterfactual_flip_rate: float
    top_features: list[FeatureInfo]
    suggestions: list[str]
    group_selection_rates: dict[str, float]
    # ----- Phase 1 additions -----
    detected_sensitive_cols: list[str]
    intersectional_metrics: list[IntersectionalMetric]
    shap_values: list[dict]          # [{feature: shap_float}, …] – one dict per row
    shap_truncated: bool             # True when dataset was sampled for SHAP speed


class MitigateResponse(BaseModel):
    before_metrics: FairnessMetrics
    after_metrics: FairnessMetrics
    mitigated_model_base64: str


class SummaryRequest(BaseModel):
    metrics: FairnessMetrics
    top_features: list[FeatureInfo]


class SummaryResponse(BaseModel):
    summary: str
