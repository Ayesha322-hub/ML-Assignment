"""Data preparation stage: reads data, cleans, creates train/test split."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from src.features import prepare_customer_features

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

seed = params.get("seed", 42)
test_size = params["split"]["test_size"]
sample_size = params["split"]["sample_size"]

data_path = "data/raw/olist_customers_dataset.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed)
else:
    df = pd.DataFrame({
        "customer_id": [f"c_{i}" for i in range(1000)],
        "customer_state": ["SP" if i % 2 == 0 else "RJ" for i in range(1000)],
        "customer_city": ["sao paulo" if i % 2 == 0 else "rio" for i in range(1000)],
        "customer_zip_code_prefix": [1000 + i for i in range(1000)]
    })

processed = prepare_customer_features(df)
train_df, test_df = train_test_split(processed, test_size=test_size, random_state=seed, stratify=processed["is_sp"])

os.makedirs("data/processed", exist_ok=True)
train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)
print(f"Data prepared successfully: {len(train_df)} train rows, {len(test_df)} test rows")
