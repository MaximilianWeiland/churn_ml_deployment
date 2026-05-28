import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)
N_TRIALS = 30
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def tune_model(X, y):
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    def objective(trial):
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 0, 5),
            reg_alpha=trial.suggest_float("reg_alpha", 0, 5),
            reg_lambda=trial.suggest_float("reg_lambda", 0, 5),
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
        )
        scores = cross_val_score(model, X, y, cv=CV, scoring="recall", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    print("Best Params:", study.best_params)
    return study.best_params
