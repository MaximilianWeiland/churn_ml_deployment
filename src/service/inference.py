import os
import sys
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(_PROJECT_ROOT)

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

# set the uri for mlflow model registry
_default_uri = f"sqlite:///{os.path.join(_PROJECT_ROOT, 'mlflow.db')}"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", _default_uri))

# load the latest registered version of the model
_client = mlflow.MlflowClient()
_versions = _client.get_latest_versions("churn-xgboost")
if not _versions:
    raise RuntimeError("No registered versions of 'churn-xgboost' found. Run the pipeline first.")
_latest = max(_versions, key=lambda v: int(v.version))
model = mlflow.sklearn.load_model(f"models:/churn-xgboost/{_latest.version}")
print(f"Loaded churn-xgboost v{_latest.version} (run {_latest.run_id})")

# load threshold and feature columns from the same run
_run = mlflow.get_run(_latest.run_id)
THRESHOLD = float(_run.data.params["threshold"])
print(f"Threshold: {THRESHOLD}")

# load feature columns from preprocessing
_artifacts_dir = mlflow.artifacts.download_artifacts(
    run_id=_latest.run_id, artifact_path="preprocessing.pkl"
)
preprocessing = joblib.load(_artifacts_dir)
FEATURE_COLS = preprocessing["feature_columns"]
print(f"Loaded {len(FEATURE_COLS)} feature columns")


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess_data(df)
    df = build_features(df)
    return df.reindex(columns=FEATURE_COLS, fill_value=0)


def predict(input_dict: dict) -> str:
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)
    proba = model.predict_proba(df_enc)[:, 1]
    result = int(proba[0] >= THRESHOLD)
    return "Likely to churn" if result == 1 else "Not likely to churn"
