import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score
    )

THRESHOLD = 0.4

def train_evaluate_model(X_train, y_train, X_test, y_test, hyperparams, threshold):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        **hyperparams,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    mlflow.log_params(hyperparams)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", auc)
    mlflow.sklearn.log_model(model, "model", registered_model_name="churn-xgboost")

    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")
    
    return model
