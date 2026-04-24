"""
One-time script to generate test fixtures for AutoBias backend tests.

Usage (from project root):
    python scripts/make_fixtures.py

Outputs:
    backend/tests/fixtures/adult_sample.csv
    backend/tests/fixtures/lr_model.pkl
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "backend", "tests", "fixtures")
SAMPLE_SIZE = 500
RANDOM_STATE = 42


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching UCI Adult dataset...")
    adult = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = adult.frame.copy()

    # Rename target column
    df = df.rename(columns={"class": "income"})

    # Drop rows with missing values
    df = df.dropna()

    # Sample
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)

    # Encode all object columns as integer codes (simple approach for fixtures)
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "adult_sample.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}  ({len(df)} rows, {len(df.columns)} columns)")
    print(f"Columns: {list(df.columns)}")

    # Train a LogisticRegression on all columns except target (sensitive col included)
    target_col = "income"
    sensitive_col = "sex"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X, y)
    print(f"Trained LogisticRegression  accuracy={clf.score(X, y):.3f}")

    pkl_path = os.path.join(OUTPUT_DIR, "lr_model.pkl")
    joblib.dump(clf, pkl_path)
    print(f"Saved model: {pkl_path}")

    # Print a quick summary for test authoring
    priv_val = df[sensitive_col].mode()[0]
    pos_label = df[target_col].max()
    print(f"\nTest parameters:")
    print(f"  target_column    = {target_col}")
    print(f"  sensitive_column = {sensitive_col}")
    print(f"  privileged_value = {priv_val}")
    print(f"  positive_label   = {pos_label}")


if __name__ == "__main__":
    main()
