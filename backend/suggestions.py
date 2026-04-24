THRESHOLDS = {
    "spd": 0.1,      # |SPD| > 0.1 is concerning
    "di": 0.8,       # DI < 0.8 fails the 4/5ths legal standard
    "eod": 0.1,      # |EOD| > 0.1 is concerning
    "flip_rate": 0.15,  # > 15% flip → sensitive attr drives predictions
    "proxy_importance": 0.05,
    "proxy_corr": 0.5,
}


def generate_suggestions(
    metrics: dict,
    flip_rate: float,
    top_features: list[dict],
) -> list[str]:
    suggestions = []

    if abs(metrics["statistical_parity_difference"]) > THRESHOLDS["spd"]:
        suggestions.append(
            "High statistical parity difference detected. Consider applying "
            "AIF360 Reweighing or Disparate Impact Remover as a preprocessing step."
        )

    if metrics["disparate_impact"] < THRESHOLDS["di"]:
        suggestions.append(
            f"Disparate impact ({metrics['disparate_impact']:.3f}) falls below the "
            "4/5ths rule (0.8). Review model selection criteria and consider "
            "post-processing with AIF360's Reject Option Classifier."
        )

    if abs(metrics.get("equal_opportunity_difference", 0.0)) > THRESHOLDS["eod"]:
        suggestions.append(
            "Unequal opportunity detected: the true positive rate differs "
            "significantly between groups. Consider Equalized Odds post-processing."
        )

    if flip_rate > THRESHOLDS["flip_rate"]:
        suggestions.append(
            f"Counterfactual flip rate is {flip_rate:.0%}: a large share of decisions "
            "change when the sensitive attribute is toggled. The model may be "
            "directly or indirectly relying on the sensitive attribute."
        )

    proxies = [
        f for f in top_features
        if f["importance"] > THRESHOLDS["proxy_importance"]
        and f["corr_with_sensitive"] > THRESHOLDS["proxy_corr"]
    ]
    for p in proxies[:3]:
        suggestions.append(
            f"Feature '{p['feature']}' has high model importance "
            f"({p['importance']:.2f}) and strong correlation with the sensitive "
            f"attribute ({p['corr_with_sensitive']:.2f}). Consider removing or "
            "transforming it to reduce proxy discrimination."
        )

    if not suggestions:
        suggestions.append(
            "No major fairness violations detected under current thresholds. "
            "Continue to monitor with diverse test sets."
        )

    return suggestions
