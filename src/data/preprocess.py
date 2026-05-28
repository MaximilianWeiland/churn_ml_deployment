import pandas as pd


def preprocess_data(df: pd.DataFrame, target_col: str = "churned") -> pd.DataFrame:
    # tidy headers
    df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace

    # drop ids if present
    for col in ["customerID", "CustomerID", "customer_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # drop monthly fees due to multicollinearity
    df = df.drop(columns=["monthly_fee"])

    # target to 0/1 if it's Yes/No
    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes": 1})

    # fill all NAs in numeric columns with 0
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df
