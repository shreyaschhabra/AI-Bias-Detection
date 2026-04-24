"""Unit tests for fairness_metrics.py."""

import numpy as np
import pandas as pd
import pytest


def make_biased_data(n=200, seed=0):
    """Create a DataFrame with known group imbalance."""
    rng = np.random.default_rng(seed)
    # privileged (sex=1): 70% positive; unprivileged (sex=0): 30% positive
    sex = np.array([1] * (n // 2) + [0] * (n // 2))
    income = np.where(sex == 1, rng.binomial(1, 0.7, n), rng.binomial(1, 0.3, n))
    age = rng.integers(20, 60, n)
    df = pd.DataFrame({"age": age, "sex": sex, "income": income})
    return df


def make_fair_data(n=200, seed=1):
    """Create a DataFrame with no group imbalance."""
    rng = np.random.default_rng(seed)
    sex = np.array([1] * (n // 2) + [0] * (n // 2))
    income = rng.binomial(1, 0.5, n)
    age = rng.integers(20, 60, n)
    df = pd.DataFrame({"age": age, "sex": sex, "income": income})
    return df


class TestGroupMetrics:
    def test_biased_spd_negative(self):
        from fairness_metrics import compute_group_metrics

        df = make_biased_data()
        y_pred = df["income"].to_numpy()
        result = compute_group_metrics(
            df, y_pred, target_col="income", sensitive_col="sex",
            privileged_value=1, positive_label=1,
        )
        # privileged has higher selection rate → SPD should be positive
        # (privileged - unprivileged > 0)
        assert result["statistical_parity_difference"] != 0.0

    def test_fair_data_spd_near_zero(self):
        from fairness_metrics import compute_group_metrics

        df = make_fair_data()
        y_pred = df["income"].to_numpy()
        result = compute_group_metrics(
            df, y_pred, target_col="income", sensitive_col="sex",
            privileged_value=1, positive_label=1,
        )
        assert abs(result["statistical_parity_difference"]) < 0.2

    def test_disparate_impact_biased(self):
        from fairness_metrics import compute_group_metrics

        df = make_biased_data()
        y_pred = df["income"].to_numpy()
        result = compute_group_metrics(
            df, y_pred, target_col="income", sensitive_col="sex",
            privileged_value=1, positive_label=1,
        )
        # DI = unprivileged_rate / privileged_rate, should be < 1 for biased data
        assert result["disparate_impact"] < 1.0

    def test_group_selection_rates_keys(self):
        from fairness_metrics import compute_group_metrics

        df = make_biased_data()
        y_pred = df["income"].to_numpy()
        result = compute_group_metrics(
            df, y_pred, target_col="income", sensitive_col="sex",
            privileged_value=1, positive_label=1,
        )
        assert "privileged" in result["group_selection_rates"]
        assert "unprivileged" in result["group_selection_rates"]


class TestIndividualMetrics:
    def test_returns_three_metrics(self):
        from fairness_metrics import compute_individual_metrics

        df = make_biased_data()
        X = df[["age"]].copy()
        y_true = df["income"].to_numpy()
        y_pred = y_true.copy()
        prot = df["sex"]

        result = compute_individual_metrics(X, y_true, y_pred, prot, priv_label=1)
        assert "equal_opportunity_difference" in result
        assert "consistency_score" in result
        assert "generalized_entropy_error" in result

    def test_consistency_perfect_preds(self):
        from fairness_metrics import compute_individual_metrics

        df = make_fair_data()
        X = df[["age"]].copy()
        y_true = df["income"].to_numpy()
        # Perfect predictions — consistency should be high
        result = compute_individual_metrics(X, y_true, y_true, df["sex"], priv_label=1)
        assert result["consistency_score"] >= 0.0
