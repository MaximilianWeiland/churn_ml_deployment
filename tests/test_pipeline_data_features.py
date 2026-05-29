import pytest
import pandas as pd

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features


# build small df once in order to pass it to all tests
@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "customer_id": ["abc-123", "def-456", "ghi-789"],
        "age": [30, 45, 22],
        "gender": ["Male", "Female", "Other"],
        "subscription_type": ["Basic", "Premium", "Standard"],
        "watch_hours": [5.0, 10.0, 2.5],
        "last_login_days": [3, 10, 1],
        "region": ["Europe", "Asia", "Africa"],
        "device": ["Laptop", "TV", "Mobile"],
        "monthly_fee": [8.99, 15.99, 13.99],
        "churned": ["No", "Yes", "No"],
        "payment_method": ["Credit Card", "PayPal", "Gift Card"],
        "number_of_profiles": [1, 3, 2],
        "avg_watch_time_per_day": [1.5, 3.0, 0.8],
        "favorite_genre": ["Action", "Drama", "Sci-Fi"],
    })


# tests for correct data loading

def test_load_data_returns_dataframe(tmp_path, raw_df):
    path = tmp_path / "test.csv"
    raw_df.to_csv(path, index=False)
    df = load_data(str(path))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == raw_df.shape

def test_load_data_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        load_data("/nonexistent/path/file.csv")


# tests for correct preprocessing

def test_preprocess_drops_customer_id(raw_df):
    result = preprocess_data(raw_df.copy(), target_col="churned")
    assert "customer_id" not in result.columns

def test_preprocess_drops_monthly_fee(raw_df):
    result = preprocess_data(raw_df.copy(), target_col="churned")
    assert "monthly_fee" not in result.columns

def test_preprocess_encodes_target_to_binary(raw_df):
    result = preprocess_data(raw_df.copy(), target_col="churned")
    assert set(result["churned"].unique()).issubset({0, 1})

def test_preprocess_fills_numeric_nans():
    df = pd.DataFrame({"age": [None, 30.0], "churned": [0, 1]})
    result = preprocess_data(df, target_col="churned")
    assert result["age"].isna().sum() == 0

def test_preprocess_strips_column_whitespace():
    df = pd.DataFrame({" age ": [30], " churned ": ["No"]})
    result = preprocess_data(df, target_col="churned")
    assert "age" in result.columns


# tests for correct feature engineering

def test_build_features_removes_categorical_columns(raw_df):
    df = preprocess_data(raw_df.copy(), target_col="churned")
    result = build_features(df, target_col="churned")
    cat_cols = [c for c in result.columns if result[c].dtype == object and c != "churned"]
    assert cat_cols == []

def test_build_features_creates_dummies_for_gender(raw_df):
    df = preprocess_data(raw_df.copy(), target_col="churned")
    result = build_features(df, target_col="churned")
    assert "gender" not in result.columns
    assert any(c.startswith("gender_") for c in result.columns)

def test_build_features_does_not_alter_target(raw_df):
    df = preprocess_data(raw_df.copy(), target_col="churned")
    result = build_features(df, target_col="churned")
    assert "churned" in result.columns
    assert set(result["churned"].unique()).issubset({0, 1})

def test_build_features_preserves_row_count(raw_df):
    df = preprocess_data(raw_df.copy(), target_col="churned")
    result = build_features(df, target_col="churned")
    assert len(result) == len(raw_df)
