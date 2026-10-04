# %% [markdown]
# # 01 - Exploratory Data Analysis (EDA)
# Customer Distribution & Location Analysis for Olist Dataset

# %%
import os
import pandas as pd
from src.features import prepare_customer_features, calculate_state_distribution

# %%
# Load data (handles both DVC pulled CSV and fallback sample)
data_path = "data/raw/olist_customers_dataset.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
else:
    print("Local CSV not found. Running with sample mock data.")
    df = pd.DataFrame({
        "customer_id": ["c1", "c2", "c3", "c4"],
        "customer_state": ["SP", "RJ", "SP", "MG"],
        "customer_city": ["sao paulo", "rio", "campinas", "bh"],
        "customer_zip_code_prefix": [1001, 20001, 13001, 30001]
    })

# %%
# State Distribution using reusable module
dist = calculate_state_distribution(df)
print("Top 5 States (%):\n", dist.head(5))

# %%
# Prepare Features
processed_df = prepare_customer_features(df)
print("Target Balance ('is_sp'):")
print(processed_df["is_sp"].value_counts(normalize=True))
