"""Evaluation stage: evaluates model, logs metrics and commit SHA."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import subprocess
import yaml
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

try:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
except Exception:
    sha = "unknown"

test_df = pd.read_csv("data/processed/test.csv")
X_test = test_df[["zip_prefix"]]
y_test = test_df["is_sp"]

model = joblib.load("models/model.joblib")
preds = model.predict(X_test)

metrics = {
    "commit_sha": sha,
    "seed": params.get("seed", 42),
    "accuracy": round(float(accuracy_score(y_test, preds)), 4),
    "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
    "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
    "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Evaluation complete. Metrics saved to metrics.json:")
print(json.dumps(metrics, indent=2))
