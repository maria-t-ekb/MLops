import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from mlflow.models import infer_signature
import joblib

def scale_frame(frame):
    df = frame.copy()
    X, y = df.drop(columns=['Survived']), df['Survived']
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X.values)
    return X_scale, y.values

def eval_metrics(actual, pred):
    acc = accuracy_score(actual, pred)
    prec = precision_score(actual, pred, zero_division=0)
    rec = recall_score(actual, pred, zero_division=0)
    return acc, prec, rec

if __name__ == "__main__":
    df = pd.read_csv("./titanic_clear.csv")
    X, Y = scale_frame(df)
    
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, random_state=42)
    
    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'l1_ratio': [0.001, 0.05, 0.01, 0.2],
        "penalty": ["l1", "l2", "elasticnet"],
        "loss": ['hinge', 'log_loss', 'modified_hubber'],
        "fit_intercept": [False, True],
    }
    
    mlflow.set_experiment("linear model titanic")
    
    with mlflow.start_run():
        clf_base = SGDClassifier(random_state=42)
        grid = GridSearchCV(clf_base, params, cv=3, n_jobs=4)
        grid.fit(X_train, y_train)
        
        best = grid.best_estimator_
        y_pred = best.predict(X_val)

        (acc, prec, rec) = eval_metrics(y_val, y_pred)
        
        mlflow.log_param("alpha", best.alpha)
        mlflow.log_param("l1_ratio", best.l1_ratio)
        mlflow.log_param("penalty", best.penalty)
        mlflow.log_param("loss", best.loss)
        mlflow.log_param("fit_intercept", best.fit_intercept)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        
        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        
        mlflow.sklearn.log_model(best, "model", signature=signature)
        
        with open("lr_titanic.pkl", "wb") as file:
            joblib.dump(best, file)

    dfruns = mlflow.search_runs()
    path2model = dfruns.sort_values("metrics.accuracy", ascending=False).iloc[0]['artifact_uri'].replace("file://", "") + '/model'
    print(path2model)
