"""Unit tests for src.features."""

import pandas as pd
from src.features import prepare_customer_features, calculate_state_distribution


def test_prepare_customer_features_creates_target():
    sample_df = pd.DataFrame({
        "customer_id": ["1", "2", "3"],
        "customer_state": ["sp", "rj", "SP "],
        "customer_city": ["SAO PAULO", "Rio de Janeiro", "Campinas"],
        "customer_zip_code_prefix": ["01001", "20000", "13000"]
    })

    processed = prepare_customer_features(sample_df)

    assert "is_sp" in processed.columns
    assert processed["is_sp"].tolist() == [1, 0, 1]
    assert processed["zip_prefix"].tolist() == [1001.0, 20000.0, 13000.0]


def test_calculate_state_distribution():
    sample_df = pd.DataFrame({
        "customer_state": ["SP", "SP", "RJ", "MG"]
    })
    dist = calculate_state_distribution(sample_df)
    assert dist["SP"] == 50.0
    assert dist["RJ"] == 25.0
