"""
benchmark_latency.py
Task 1 – Measure round-trip API latency for /api/analyze with 50,000-row dataset.
"""
import io
import time
import json
import sys

import numpy as np
import pandas as pd
import joblib
import requests
from sklearn.ensemble import RandomForestClassifier

ENDPOINT = "http://localhost:8000/api/analyze"
N_ROWS   = 50_000
N_RUNS   = 3          # warm + 2 measured runs; we report the median

# ── 1. Build a realistic 50k-row synthetic dataset ─────────────────────────
rng = np.random.default_rng(42)

# gender encoded as int (1=Male, 0=Female) so sklearn can ingest it directly.
# The endpoint will pass the raw CSV column to the model as-is (X.to_numpy()).
gender_int = rng.integers(0, 2, size=N_ROWS)   # 1=Male, 0=Female

df = pd.DataFrame({
    "age":         rng.integers(18, 70,  size=N_ROWS),
    "income":      rng.integers(20000, 120000, size=N_ROWS),
    "credit_score":rng.integers(300, 850, size=N_ROWS),
    "loan_amount": rng.integers(1000, 50000, size=N_ROWS),
    "gender":      gender_int,          # 1=Male privileged, 0=Female unprivileged
    "race":        rng.integers(0, 4, size=N_ROWS),  # encoded 0-3
    "employed":    rng.integers(0, 2, size=N_ROWS),
    "education":   rng.integers(1, 5, size=N_ROWS),
    "approved":    rng.integers(0, 2, size=N_ROWS),   # target
})

print(f"[benchmark_latency] Dataset shape: {df.shape}")

# ── 2. Train a quick RF on the same data so the .pkl is valid ──────────────
# Train on ALL non-target columns — matching what the endpoint will see
feature_cols = [c for c in df.columns if c != "approved"]
X_train = df[feature_cols].copy()
y_train = df["approved"].to_numpy()

clf = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Serialize model to bytes
model_buf = io.BytesIO()
joblib.dump(clf, model_buf)
model_buf.seek(0)
model_bytes = model_buf.read()

# Serialize CSV to bytes
csv_buf = io.BytesIO()
df.to_csv(csv_buf, index=False)
csv_buf.seek(0)
csv_bytes = csv_buf.read()

print(f"[benchmark_latency] CSV payload size : {len(csv_bytes)/1_000_000:.2f} MB")
print(f"[benchmark_latency] Model payload size: {len(model_bytes)/1_000:.1f} KB")

# ── 3. Send N_RUNS requests and record wall-clock latency ──────────────────
latencies_ms = []

for run in range(N_RUNS):
    files = {
        "file":  ("dataset.csv",      io.BytesIO(csv_bytes),   "text/csv"),
        "model": ("model.pkl",         io.BytesIO(model_bytes), "application/octet-stream"),
    }
    data = {
        "target_column":    "approved",
        "sensitive_column": "gender",
        "privileged_value": "1",        # 1=Male (integer-encoded)
        "positive_label":   "1",
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(ENDPOINT, files=files, data=data, timeout=300)
        t1 = time.perf_counter()
    except Exception as e:
        print(f"  [run {run}] REQUEST FAILED: {e}")
        sys.exit(1)

    elapsed_ms = (t1 - t0) * 1000
    latencies_ms.append(elapsed_ms)

    if resp.status_code != 200:
        print(f"  [run {run}] HTTP {resp.status_code}: {resp.text[:300]}")
        sys.exit(1)

    payload = resp.json()
    shap_count = len(payload.get("shap_values", []))
    print(f"  [run {run+1}] latency={elapsed_ms:,.0f} ms | SHAP rows returned={shap_count}")

# ── 4. Compute SHAP JSON payload size ──────────────────────────────────────
# Re-use the last response
shap_json_bytes = json.dumps(payload.get("shap_values", [])).encode("utf-8")
shap_mb = len(shap_json_bytes) / 1_000_000

median_ms  = sorted(latencies_ms)[len(latencies_ms) // 2]
min_ms     = min(latencies_ms)
max_ms     = max(latencies_ms)

# ── 5. Output results ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("  TASK 1 RESULTS – API LATENCY (50,000-row dataset)")
print("="*60)
print(f"  Runs            : {N_RUNS}")
print(f"  Latency (min)   : {min_ms:>10,.0f} ms")
print(f"  Latency (median): {median_ms:>10,.0f} ms   ← USE THIS")
print(f"  Latency (max)   : {max_ms:>10,.0f} ms")
print(f"  SHAP JSON size  : {shap_mb:>10.3f} MB  (returned to frontend)")
print(f"  SHAP rows       : {shap_count:>10,}")
print("="*60)

# Write a machine-readable result for the summary script
with open("/tmp/bench_latency_result.json", "w") as f:
    json.dump({
        "median_ms":  round(median_ms),
        "min_ms":     round(min_ms),
        "max_ms":     round(max_ms),
        "shap_mb":    round(shap_mb, 3),
        "shap_rows":  shap_count,
        "n_rows":     N_ROWS,
    }, f)

print("[benchmark_latency] Done. Results saved to /tmp/bench_latency_result.json")
