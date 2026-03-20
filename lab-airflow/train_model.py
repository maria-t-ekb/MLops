import os
import json
from pathlib import Path

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib

from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature

AIRFLOW_HOME = Path(os.environ.get("AIRFLOW_HOME", ".")).resolve()
DATA_DIR = AIRFLOW_HOME / "data" / "climate_energy"
ARTIFACTS_DIR = AIRFLOW_HOME / "artifacts" / "climate_energy"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def scale_frame(frame):
    df = frame.copy()
    X, y = df.drop(columns=["avg_temperature"]), df["avg_temperature"]

    scaler = StandardScaler()
    power_trans = PowerTransformer()

    X_scale = scaler.fit_transform(X.values)
    Y_scale = power_trans.fit_transform(y.values.reshape(-1, 1))

    return X_scale, Y_scale, scaler, power_trans

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train():
    df = pd.read_csv(DATA_DIR / "df_clear.csv")

    X, Y, scaler, power_trans = scale_frame(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        Y,
        test_size=0.3,
        random_state=42,
    )

    params = {
        "alpha": [0.0001, 0.001, 0.01, 0.05, 0.1],
        "l1_ratio": [0.001, 0.05, 0.01, 0.2],
        "penalty": ["l1", "l2", "elasticnet"],
        "loss": ["squared_error", "huber", "epsilon_insensitive"],
        "fit_intercept": [False, True],
    }

    mlflow.set_tracking_uri(f"file://{(AIRFLOW_HOME / 'mlruns').as_posix()}")
    mlflow.set_experiment("linear model climate energy")

    with mlflow.start_run():
        lr = SGDRegressor(random_state=42, max_iter=2000, tol=1e-3)
        clf = GridSearchCV(lr, params, cv=3, n_jobs=1)
        clf.fit(X_train, y_train.reshape(-1))

        best = clf.best_estimator_

        y_pred = best.predict(X_val)
        y_pred_real = power_trans.inverse_transform(y_pred.reshape(-1, 1))
        y_val_real = power_trans.inverse_transform(y_val)

        rmse, mae, r2 = eval_metrics(y_val_real, y_pred_real)

        mlflow.log_param("alpha", best.alpha)
        mlflow.log_param("l1_ratio", best.l1_ratio)
        mlflow.log_param("penalty", best.penalty)
        mlflow.log_param("eta0", best.eta0)
        mlflow.log_param("loss", best.loss)
        mlflow.log_param("fit_intercept", best.fit_intercept)
        mlflow.log_param("epsilon", best.epsilon)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)

        with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "r2": float(r2),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        joblib.dump(best, ARTIFACTS_DIR / "climate_energy_model.pkl")
        joblib.dump(scaler, ARTIFACTS_DIR / "scaler.pkl")
        joblib.dump(power_trans, ARTIFACTS_DIR / "power_transformer.pkl")

        print(f"[OK] Saved model to: {ARTIFACTS_DIR / 'climate_energy_model.pkl'}")
        print(f"[OK] Saved metrics to: {ARTIFACTS_DIR / 'metrics.json'}")
        print(f"[OK] RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }
