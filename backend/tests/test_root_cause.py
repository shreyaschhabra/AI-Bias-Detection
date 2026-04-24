"""Unit tests for root_cause.py."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


def make_wrapper_with_importances(feature_names, importances):
    wrapper = MagicMock()
    wrapper.feature_importances.return_value = dict(zip(feature_names, importances))
    wrapper.model = MagicMock()
    return wrapper


def make_wrapper_no_importances():
    wrapper = MagicMock()
    wrapper.feature_importances.return_value = {}
    # For permutation_importance fallback we need a real model
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=200)
    wrapper.model = clf
    return wrapper


class TestRootCause:
    def test_returns_top_k_features(self):
        from root_cause import compute_root_cause

        n = 100
        df = pd.DataFrame({
            "age": np.random.randint(20, 60, n),
            "income": np.random.rand(n),
            "zip": np.random.randint(0, 100, n),
            "sex": np.random.randint(0, 2, n),
        })
        y_true = np.random.randint(0, 2, n)
        wrapper = make_wrapper_with_importances(
            ["age", "income", "zip"], [0.5, 0.3, 0.2]
        )

        result = compute_root_cause(wrapper, df, y_true, sensitive_col="sex", top_k=2)
        assert len(result) == 2
        assert result[0]["feature"] == "age"  # highest importance first

    def test_result_has_required_keys(self):
        from root_cause import compute_root_cause

        n = 50
        df = pd.DataFrame({
            "a": np.random.rand(n),
            "sex": np.random.randint(0, 2, n),
        })
        y_true = np.random.randint(0, 2, n)
        wrapper = make_wrapper_with_importances(["a"], [1.0])

        result = compute_root_cause(wrapper, df, y_true, sensitive_col="sex")
        assert len(result) >= 1
        item = result[0]
        assert "feature" in item
        assert "importance" in item
        assert "corr_with_sensitive" in item

    def test_proxy_feature_gets_high_corr(self):
        """A feature identical to the sensitive column should have corr_with_sensitive ≈ 1."""
        from root_cause import compute_root_cause

        n = 200
        sex = np.random.randint(0, 2, n)
        df = pd.DataFrame({
            "proxy": sex,          # perfectly correlated with sensitive
            "noise": np.random.rand(n),
            "sex": sex,
        })
        y_true = np.random.randint(0, 2, n)
        wrapper = make_wrapper_with_importances(["proxy", "noise"], [0.6, 0.4])

        result = compute_root_cause(wrapper, df, y_true, sensitive_col="sex")
        proxy_entry = next(r for r in result if r["feature"] == "proxy")
        assert proxy_entry["corr_with_sensitive"] > 0.8
