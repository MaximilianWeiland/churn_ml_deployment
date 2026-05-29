import os
import sys
import time
import argparse
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.load_data import load_data
from src.data.validate_data import validate_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.model.tune import tune_model
from src.model.train_evaluate import train_evaluate_model



def main(args):
    """
    Main training pipeline function.
    """
    
    # configure MLflow
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    _default_uri = f"sqlite:///{os.path.join(project_root, 'mlflow.db')}"
    mlruns_path = args.mlflow_uri or os.getenv("MLFLOW_TRACKING_URI") or _default_uri
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)

    # start the mlflow run
    with mlflow.start_run():

        # log hyperparams and configuration
        mlflow.log_param("model", args.model_name)
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("test_size", args.test_size)

        # load the data
        print("Loading data...")
        df = load_data(args.input)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # validate the data with great expectations
        print("Validating data quality with Great Expectations...")
        is_valid, failed = validate_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))

        # log errors if validation failed
        if not is_valid:
            import json
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"Data quality check failed. Issues: {failed}")
        else:
            print("Data validation passed. Logged to MLflow.")

        # preprocess the data
        print("Preprocessing data...")
        df = preprocess_data(df)

        # save processed dataset for reproducibility and debugging
        processed_path = os.path.join(project_root, "data", "processed", "netflix_churn_processed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"Processed dataset saved to {processed_path} | Shape: {df.shape}")

        # apply feature engineering
        print("Building features...")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        df_enc = build_features(df, target_col=target)
        print(f"Feature engineering completed: {df_enc.shape[1]} features")

        # save feature metadata for serving consistency both locally and to MLflow
        import json, joblib
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        feature_cols = list(df_enc.drop(columns=[target]).columns)
        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")

        # save preprocessing artifacts for serving pipeline
        preprocessing_artifact = {
            "feature_columns": feature_cols,
            "target": target
        }
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
        print(f"Saved {len(feature_cols)} feature columns for serving consistency")

        # split the data into training and test set (stratified split)
        print("Splitting data...")
        X = df_enc.drop(columns=[target])
        y = df_enc[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=args.test_size,
            stratify=y,
            random_state=42
        )
        print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")


        # apply hyperparameter tuning via cross-validation
        print("Hyperparameter tuning...")
        best_params = tune_model(X_train, y_train)

        
        # train the model with the optimized hyperparameters
        print("Training XGBoost model...")
        t0 = time.time()
        _ = train_evaluate_model(X_train, y_train, X_test, y_test, best_params, args.threshold)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)
        print(f"Model trained in {train_time:.2f} seconds")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run churn pipeline with XGBoost + MLflow")
    p.add_argument("--input", type=str, required=True,
                   help="path to CSV (e.g., data/raw/netflix_customer_churn.csv)")
    p.add_argument("--target", type=str, default="churned")
    p.add_argument("--model_name", type=str, default="xgboost")
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Netflix Churn")
    p.add_argument("--mlflow_uri", type=str, default=None,
                    help="override MLflow tracking URI, else uses project_root/mlruns")

    args = p.parse_args()
    main(args)

"""
# Use this below to run the pipeline:

python scripts/run_pipeline.py \                                            
    --input data/raw/netflix_customer_churn.csv \
    --target Churn
"""
