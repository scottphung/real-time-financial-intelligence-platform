import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processing.ml.build_dataset import build_dataset


def make_sample_df(n=20):
    """Create a minimal valid dataframe for testing."""
    prices = [45000 + i * 10 for i in range(n)]
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="5s"),
        "price": prices,
        "return": [0.0] + [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, n)],
        "moving_avg_10": prices,
        "volatility": [0.0] * n,
        "is_anomaly": [False] * n
    })


def test_build_dataset_returns_correct_shape():
    df = make_sample_df(20)
    X, y = build_dataset(df)
    assert X is not None
    assert y is not None
    assert len(X) == len(y)
    assert len(X) > 0


def test_build_dataset_correct_features():
    df = make_sample_df(20)
    X, y = build_dataset(df)
    expected_features = ["price", "return", "moving_avg_10", "volatility", "is_anomaly"]
    for feat in expected_features:
        assert feat in X.columns, f"Missing feature: {feat}"


def test_build_dataset_no_nulls():
    df = make_sample_df(20)
    X, y = build_dataset(df)
    assert X.isnull().sum().sum() == 0, "X contains null values"
    assert y.isnull().sum() == 0, "y contains null values"


def test_build_dataset_binary_target():
    df = make_sample_df(20)
    X, y = build_dataset(df)
    assert set(y.unique()).issubset({0, 1}), "Target must be binary (0 or 1)"


def test_build_dataset_empty_input():
    X, y = build_dataset(None)
    assert X is None
    assert y is None


def test_build_dataset_insufficient_rows():
    df = make_sample_df(3)
    X, y = build_dataset(df)
    assert X is None or len(X) < 10