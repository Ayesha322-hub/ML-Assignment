"""Feature extraction and data preparation functions."""

import numpy as np
import pandas as pd


def prepare_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess customer data and create target for binary classification.
    Target: 1 if customer is in Sao Paulo ('SP'), 0 otherwise.
    Features: zip_prefix, is_capital, state.
    """
    data = df.copy()

    # Clean text columns
    if "customer_state" in data.columns:
        data["customer_state"] = data["customer_state"].astype(str).str.strip().str.upper()
    if "customer_city" in data.columns:
        data["customer_city"] = data["customer_city"].astype(str).str.strip().str.lower()

    # Create Binary Target: Top hub 'SP' vs Others
    data["is_sp"] = (data["customer_state"] == "SP").astype(int)

    # Zip code feature
    if "customer_zip_code_prefix" in data.columns:
        data["zip_prefix"] = pd.to_numeric(data["customer_zip_code_prefix"], errors="coerce").fillna(0)
    else:
        data["zip_prefix"] = 0

    return data


def calculate_state_distribution(df: pd.DataFrame) -> pd.Series:
    """Calculate percentage share of each state."""
    if "customer_state" not in df.columns:
        return pd.Series(dtype=float)
    return df["customer_state"].value_counts(normalize=True) * 100
