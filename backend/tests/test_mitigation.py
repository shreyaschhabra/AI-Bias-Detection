"""Integration tests for POST /api/mitigate endpoint."""

import base64
import io
import os

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from fastapi.testclient import TestClient


def _make_biased_dataset(n=200, seed=42):
    """Create a biased dataset where 'sex' (sensitive col) heavily dictates 'income'."""
    rng = np.random.default_rng(seed)
    sex = np.array([1] * (n // 2) + [0] * (n // 2))  # 1: privileged, 0: unprivileged
    age = rng.integers(20, 60, n)
    # Income naturally higher for sex = 1 purely due to bias
    income = np.where(sex == 1, rng.binomial(1, 0.9, n), rng.binomial(1, 0.1, n))
    return pd.DataFrame({"age": age, "sex": sex, "income": income})


def _train_lr(df, feature_cols, target_col):
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(df[feature_cols].to_numpy(), df[target_col].to_numpy())
    return clf


@pytest.fixture(scope="module")
def client():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from main import app
    return TestClient(app)


@pytest.fixture(scope="module")
def biased_files():
    df = _make_biased_dataset()
    feature_cols = [c for c in df.columns if c != "income"]
    clf = _train_lr(df, feature_cols, "income")

    csv_buf = io.BytesIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    pkl_buf = io.BytesIO()
    joblib.dump(clf, pkl_buf)
    pkl_buf.seek(0)

    return csv_buf, pkl_buf


class TestMitigateEndpoint:
    def test_mitigate_endpoint_improves_fairness(self, client, biased_files):
        csv_buf, pkl_buf = biased_files
        csv_buf.seek(0)
        pkl_buf.seek(0)

        resp = client.post(
            "/api/mitigate",
            files={
                "file": ("data.csv", csv_buf, "text/csv"),
                "model": ("model.pkl", pkl_buf, "application/octet-stream"),
            },
            data={
                "target_column": "income",
                "sensitive_column": "sex",
                "privileged_value": "1",
                "positive_label": "1",
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

        assert "before_metrics" in body
        assert "after_metrics" in body
        assert "mitigated_model_base64" in body

        # The biased model should have very severe Disparate Impact
        di_before = body["before_metrics"]["disparate_impact"]
        di_after = body["after_metrics"]["disparate_impact"]
        
        # Reweighing should generally bring DI closer to 1.0 (or at least out of extreme bias bounds)
        # Because we constructed an extreme 90/10 split, DI before will be tiny.
        assert di_before < 0.3
        
        # Noticeably improved disparity Impact
        assert di_after > di_before

        # Ensure the model is interpretable and deserializable
        b64_str = body["mitigated_model_base64"]
        b64_bytes = base64.b64decode(b64_str)
        buffer = io.BytesIO(b64_bytes)
        new_model = joblib.load(buffer)
        
        # Ensure it works
        assert hasattr(new_model, "predict")
