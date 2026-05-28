import pandas as pd


def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    df = df.copy()

    # identify all categorical and boolean columns
    cat_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    bool_cols = [c for c in df.select_dtypes(include='bool').columns if c != target_col]

    # one-hot encode all categorical columns as all of them have more than two values
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # convert booleans to int
    df[bool_cols] = df[bool_cols].astype(int)

    return df

