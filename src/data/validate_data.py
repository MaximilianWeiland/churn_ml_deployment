import pandas as pd
from typing import Tuple, List


REQUIRED_COLUMNS = [
    "customer_id", "age", "gender", "subscription_type", "watch_hours",
    "last_login_days", "region", "device", "monthly_fee", "payment_method",
    "number_of_profiles", "avg_watch_time_per_day", "favorite_genre", "churned",
]

VALID_VALUES = {
    "churned": {0, 1},
    "gender": {"Male", "Female", "Other"},
    "subscription_type": {"Basic", "Standard", "Premium"},
    "region": {"Africa", "Europe", "Asia", "Oceania", "South America", "North America"},
    "device": {"TV", "Mobile", "Laptop", "Desktop", "Tablet"},
    "payment_method": {"Gift Card", "Crypto", "Debit Card", "PayPal", "Credit Card"},
    "favorite_genre": {"Action", "Sci-Fi", "Drama", "Horror", "Romance", "Comedy", "Documentary"},
}

NUMERIC_RANGES = {
    "age": (12, 120),
    "watch_hours": (0, None),
    "last_login_days": (0, None),
    "monthly_fee": (0, 20),
    "number_of_profiles": (1, 5),
    "avg_watch_time_per_day": (0, 500),
}


def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    print("Starting data validation...")
    failed: List[str] = []

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            failed.append(f"expect_column_to_exist: {col}")

    for col, valid_set in VALID_VALUES.items():
        if col in df.columns and not set(df[col].dropna().unique()).issubset(valid_set):
            failed.append(f"expect_column_values_to_be_in_set: {col}")

    for col, (min_val, max_val) in NUMERIC_RANGES.items():
        if col not in df.columns:
            continue
        if min_val is not None and (df[col] < min_val).any():
            failed.append(f"expect_column_values_to_be_between (min): {col}")
        if max_val is not None and (df[col] > max_val).any():
            failed.append(f"expect_column_values_to_be_between (max): {col}")

    passed = len(failed) == 0
    total = len(REQUIRED_COLUMNS) + len(VALID_VALUES) + len(NUMERIC_RANGES)
    if passed:
        print(f"Data validation PASSED: {total}/{total} checks successful")
    else:
        print(f"Data validation FAILED: {len(failed)}/{total} checks failed")
        print(f"Failed expectations: {failed}")

    return passed, failed
