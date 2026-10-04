"""Training stage: trains Random Forest model and saves checkpoint."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

seed = params.get("seed", 42)
n_estimators = params["train"]["n_estimators"]
max_depth = params["train"]["max_depth"]

train_df = pd.read_csv("data/processed/train.csv")
X_train = train_df[["zip_prefix"]]
y_train = train_df["is_sp"]

model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")
print(f"Model trained and saved to models/model.joblib (trees={n_estimators}, depth={max_depth})")
