"""Integration tests for POST /analyze endpoint."""

import io
import os

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

# Fixtures dir
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _make_simple_dataset(n=100, seed=0):
    """Create a minimal biased dataset for quick tests (no file I/O)."""
    rng = np.random.default_rng(seed)
    sex = np.array([1] * (n // 2) + [0] * (n // 2))
    age = rng.integers(20, 60, n)
    income = np.where(sex == 1, rng.binomial(1, 0.7, n), rng.binomial(1, 0.3, n))
    return pd.DataFrame({"age": age, "sex": sex, "income": income})


def _train_lr(df, feature_cols, target_col):
    clf = LogisticRegression(max_iter=500, random_state=0)
    clf.fit(df[feature_cols].to_numpy(), df[target_col].to_numpy())
    return clf


@pytest.fixture(scope="module")
def client():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from main import app
    return TestClient(app)


@pytest.fixture(scope="module")
def simple_files():
    df = _make_simple_dataset()
    # Train on all feature cols (age + sex) — matching the endpoint's feature_cols logic
    feature_cols = [c for c in df.columns if c != "income"]
    clf = _train_lr(df, feature_cols, "income")

    csv_buf = io.BytesIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    pkl_buf = io.BytesIO()
    joblib.dump(clf, pkl_buf)
    pkl_buf.seek(0)

    return csv_buf, pkl_buf


class TestAnalyzeEndpoint:
    def test_status_ok(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_golden_path(self, client, simple_files):
        csv_buf, pkl_buf = simple_files
        csv_buf.seek(0)
        pkl_buf.seek(0)

        resp = client.post(
            "/analyze",
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
        assert "metrics" in body
        assert "counterfactual_flip_rate" in body
        assert "top_features" in body
        assert "suggestions" in body
        assert "group_selection_rates" in body
        assert isinstance(body["counterfactual_flip_rate"], float)
        assert 0.0 <= body["counterfactual_flip_rate"] <= 1.0

    def test_wrong_target_column_returns_400(self, client, simple_files):
        csv_buf, pkl_buf = simple_files
        csv_buf.seek(0)
        pkl_buf.seek(0)

        resp = client.post(
            "/analyze",
            files={
                "file": ("data.csv", csv_buf, "text/csv"),
                "model": ("model.pkl", pkl_buf, "application/octet-stream"),
            },
            data={
                "target_column": "nonexistent_col",
                "sensitive_column": "sex",
                "privileged_value": "1",
            },
        )
        assert resp.status_code == 400
        assert "nonexistent_col" in resp.json()["detail"]

    def test_invalid_model_file_returns_400(self, client, simple_files):
        csv_buf, _ = simple_files
        csv_buf.seek(0)

        resp = client.post(
            "/analyze",
            files={
                "file": ("data.csv", csv_buf, "text/csv"),
                "model": ("model.txt", io.BytesIO(b"not a model"), "text/plain"),
            },
            data={
                "target_column": "income",
                "sensitive_column": "sex",
                "privileged_value": "1",
            },
        )
        assert resp.status_code == 400

    def test_metrics_structure(self, client, simple_files):
        csv_buf, pkl_buf = simple_files
        csv_buf.seek(0)
        pkl_buf.seek(0)

        resp = client.post(
            "/analyze",
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
        assert resp.status_code == 200
        metrics = resp.json()["metrics"]
        for key in [
            "statistical_parity_difference",
            "disparate_impact",
            "equal_opportunity_difference",
            "consistency_score",
            "generalized_entropy_error",
        ]:
            assert key in metrics, f"Missing metric: {key}"
            assert isinstance(metrics[key], float)
