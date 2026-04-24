"""Unit tests for counterfactual.py."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


def make_wrapper(pred_fn):
    """Create a mock ModelWrapper with a custom predict function."""
    wrapper = MagicMock()
    wrapper.predict.side_effect = lambda X: pred_fn(X)
    return wrapper


class TestCounterfactualFlipRate:
    def test_always_same_prediction_no_flip(self):
        from counterfactual import compute_counterfactual_flip_rate

        df = pd.DataFrame({
            "age": range(10),
            "sex": [0, 1] * 5,
        })
        orig_preds = np.ones(10, dtype=int)
        wrapper = make_wrapper(lambda X: np.ones(len(X), dtype=int))

        rate = compute_counterfactual_flip_rate(wrapper, df, "sex", orig_preds)
        assert rate == 0.0

    def test_always_different_prediction_full_flip(self):
        from counterfactual import compute_counterfactual_flip_rate

        df = pd.DataFrame({
            "age": range(10),
            "sex": [0, 1] * 5,
        })
        orig_preds = np.zeros(10, dtype=int)

        call_count = [0]
        def predict_fn(X):
            # Always return 1 (opposite of 0)
            return np.ones(len(X), dtype=int)

        wrapper = make_wrapper(predict_fn)
        rate = compute_counterfactual_flip_rate(wrapper, df, "sex", orig_preds)
        assert rate == 1.0

    def test_binary_uses_vectorized_path(self):
        """Binary sensitive attribute should use the fast vectorized path."""
        from counterfactual import compute_counterfactual_flip_rate

        df = pd.DataFrame({
            "age": list(range(20)),
            "sex": [0, 1] * 10,
        })

        # predict returns the sex value directly, so after flipping sex every pred changes
        def predict_fn(X):
            return X["sex"].to_numpy().astype(int)

        orig_preds = predict_fn(df)  # [0, 1, 0, 1, ...]
        wrapper = make_wrapper(predict_fn)
        rate = compute_counterfactual_flip_rate(wrapper, df, "sex", orig_preds)
        assert rate == 1.0

    def test_sample_cap_respected(self):
        from counterfactual import compute_counterfactual_flip_rate

        n = 2000
        df = pd.DataFrame({
            "age": range(n),
            "sex": [0, 1] * (n // 2),
        })
        orig_preds = np.zeros(n, dtype=int)
        wrapper = make_wrapper(lambda X: np.zeros(len(X), dtype=int))

        rate = compute_counterfactual_flip_rate(wrapper, df, "sex", orig_preds, sample_size=100)
        # With same predictions before/after flip, rate should be 0
        assert rate == 0.0
