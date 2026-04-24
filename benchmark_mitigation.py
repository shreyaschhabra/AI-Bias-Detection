"""
benchmark_mitigation.py
Task 2 – Quantify bias reduction via /api/mitigate (AIF360 Reweighing).

Strategy:
  • Generate a mathematically biased dataset where a protected attribute
    (gender=Male) is ~4× more likely to receive a positive outcome.
  • Hit /api/analyze  → record baseline Disparate Impact (DI) & SPD.
  • Hit /api/mitigate → record post-mitigation DI & SPD.
  • Compute % improvement toward fairness (DI toward 1.0, SPD toward 0.0).
"""
import io
import json
import sys

import numpy as np
import pandas as pd
import joblib
import requests
from sklearn.linear_model import LogisticRegression

ANALYZE_URL  = "http://localhost:8000/api/analyze"
MITIGATE_URL = "http://localhost:8000/api/mitigate"
N_ROWS       = 5_000          # enough for stable AIF360 metrics, fast turnaround

rng = np.random.default_rng(0)

# ── 1. Build a deliberately biased dataset ────────────────────────────────
#   Male   → approved with p=0.75  (privileged)
#   Female → approved with p=0.20  (unprivileged)
#   DI_theoretical = 0.20 / 0.75 ≈ 0.267  (well below 0.8 fairness threshold)

# gender encoded as int: 1=Male (privileged), 0=Female (unprivileged)
# This keeps all features numeric so sklearn can ingest the CSV directly.
gender_int  = rng.choice([1, 0], size=N_ROWS, p=[0.5, 0.5])  # 1=Male, 0=Female
age         = rng.integers(22, 60, size=N_ROWS)
income      = rng.integers(30_000, 100_000, size=N_ROWS)
score       = rng.integers(400, 800, size=N_ROWS)
edu         = rng.integers(1, 5, size=N_ROWS)

approved = np.where(
    gender_int == 1,
    rng.binomial(1, 0.75, size=N_ROWS),   # Male: 75% approval
    rng.binomial(1, 0.20, size=N_ROWS),   # Female: 20% approval
)

df = pd.DataFrame({
    "gender":      gender_int,    # 1=Male
    "age":         age,
    "income":      income,
    "credit_score": score,
    "education":   edu,
    "approved":    approved,
})

print(f"[benchmark_mitigation] Dataset shape : {df.shape}")
male_rate   = approved[gender_int == 1].mean()
female_rate = approved[gender_int == 0].mean()
theoretical_di  = female_rate / male_rate
theoretical_spd = female_rate - male_rate
print(f"[benchmark_mitigation] Male approval rate   : {male_rate:.3f}")
print(f"[benchmark_mitigation] Female approval rate : {female_rate:.3f}")
print(f"[benchmark_mitigation] Theoretical DI       : {theoretical_di:.3f}")
print(f"[benchmark_mitigation] Theoretical SPD      : {theoretical_spd:.3f}")

# ── 2. Train a LR model on ALL non-target columns (all numeric, no OHE needed)
feature_cols_raw = [c for c in df.columns if c != "approved"]
clf_raw = LogisticRegression(max_iter=500, random_state=42)
clf_raw.fit(df[feature_cols_raw], df["approved"])
model_buf = io.BytesIO()
joblib.dump(clf_raw, model_buf)
model_buf.seek(0)
model_bytes = model_buf.read()

# Serialize CSV to bytes (must happen before post_analyze / post_mitigate)
csv_bytes = df.to_csv(index=False).encode("utf-8")
print(f"[benchmark_mitigation] CSV payload size : {len(csv_bytes)/1_000_000:.2f} MB")

# ── 3. /api/analyze – baseline metrics ───────────────────────────────────
def post_analyze():
    files = {
        "file":  ("dataset.csv", io.BytesIO(csv_bytes),   "text/csv"),
        "model": ("model.pkl",   io.BytesIO(model_bytes), "application/octet-stream"),
    }
    data = {
        "target_column":    "approved",
        "sensitive_column": "gender",
        "privileged_value": "1",     # 1=Male (integer-encoded)
        "positive_label":   "1",
    }
    r = requests.post(ANALYZE_URL, files=files, data=data, timeout=300)
    if r.status_code != 200:
        print(f"  /api/analyze error {r.status_code}: {r.text[:400]}")
        sys.exit(1)
    return r.json()

print("\n[benchmark_mitigation] Calling /api/analyze for BASELINE metrics …")
baseline = post_analyze()
b_di  = baseline["metrics"]["disparate_impact"]
b_spd = baseline["metrics"]["statistical_parity_difference"]
print(f"  BASELINE Disparate Impact : {b_di:.4f}")
print(f"  BASELINE Stat Parity Diff : {b_spd:.4f}")

# ── 4. /api/mitigate – post-mitigation metrics ────────────────────────────
def post_mitigate():
    files = {
        "file":  ("dataset.csv", io.BytesIO(csv_bytes),   "text/csv"),
        "model": ("model.pkl",   io.BytesIO(model_bytes), "application/octet-stream"),
    }
    data = {
        "target_column":    "approved",
        "sensitive_column": "gender",
        "privileged_value": "1",     # 1=Male (integer-encoded)
        "positive_label":   "1",
    }
    r = requests.post(MITIGATE_URL, files=files, data=data, timeout=300)
    if r.status_code != 200:
        print(f"  /api/mitigate error {r.status_code}: {r.text[:400]}")
        sys.exit(1)
    return r.json()

print("[benchmark_mitigation] Calling /api/mitigate (Reweighing + retrain) …")
mitigated = post_mitigate()
a_di  = mitigated["after_metrics"]["disparate_impact"]
a_spd = mitigated["after_metrics"]["statistical_parity_difference"]
print(f"  AFTER   Disparate Impact  : {a_di:.4f}")
print(f"  AFTER   Stat Parity Diff  : {a_spd:.4f}")

# ── 5. Calculate improvement ───────────────────────────────────────────────
# DI improvement: how much closer to 1.0 (perfect fairness) did we get?
#   deviation_before = |1 - b_di|   deviation_after = |1 - a_di|
#   pct_improvement  = (deviation_before - deviation_after) / deviation_before * 100
di_dev_before  = abs(1.0 - b_di)
di_dev_after   = abs(1.0 - a_di)
di_improvement_pct = ((di_dev_before - di_dev_after) / di_dev_before) * 100 if di_dev_before > 0 else 0.0

# SPD improvement: how much closer to 0.0?
spd_dev_before  = abs(b_spd)
spd_dev_after   = abs(a_spd)
spd_improvement_pct = ((spd_dev_before - spd_dev_after) / spd_dev_before) * 100 if spd_dev_before > 0 else 0.0

print("\n" + "="*60)
print("  TASK 2 RESULTS – BIAS REDUCTION (Reweighing)")
print("="*60)
print(f"  Metric : Disparate Impact (DI)")
print(f"    Before : {b_di:.4f}  (deviation from 1.0 = {di_dev_before:.4f})")
print(f"    After  : {a_di:.4f}  (deviation from 1.0 = {di_dev_after:.4f})")
print(f"    ✓ DI improved toward fairness by  {di_improvement_pct:+.1f}%")
print(f"  Metric : Statistical Parity Difference (SPD)")
print(f"    Before : {b_spd:.4f}")
print(f"    After  : {a_spd:.4f}")
print(f"    ✓ SPD improved toward 0 by         {spd_improvement_pct:+.1f}%")
print("="*60)

with open("/tmp/bench_mitigation_result.json", "w") as f:
    json.dump({
        "baseline_di":         round(b_di, 4),
        "after_di":            round(a_di, 4),
        "di_improvement_pct":  round(di_improvement_pct, 1),
        "baseline_spd":        round(b_spd, 4),
        "after_spd":           round(a_spd, 4),
        "spd_improvement_pct": round(spd_improvement_pct, 1),
    }, f)

print("[benchmark_mitigation] Done. Results saved to /tmp/bench_mitigation_result.json")
