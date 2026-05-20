import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Titanic_Survival_Prediction")

def preprocessing_titanic(frame):
    df = frame.copy()

    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    cat_columns = ["Sex", "Embarked"]
    ordinal = OrdinalEncoder()
    df[cat_columns] = ordinal.fit_transform(df[cat_columns])

    return df

def scale_titanic(frame):
    df = frame.copy()
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X)

    return X_scale, y.values

def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, zero_division=0)
    recall = recall_score(actual, pred, zero_division=0)
    return accuracy, precision, recall

df = pd.read_csv('titanic.csv')

df_proc = preprocessing_titanic(df)
X, Y = scale_titanic(df_proc)

X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y
)


params_sgd = {
    "alpha": [0.0001, 0.001, 0.01, 0.1],
    "loss": ["hinge", "log_loss"],
}

with mlflow.start_run(run_name="SGD_Classifier_GridSearch"):
    clf_sgd = SGDClassifier(random_state=42)
    grid = GridSearchCV(clf_sgd, params_sgd, cv=5)
    grid.fit(X_train, y_train)

    best_sgd = grid.best_estimator_
    y_pred = best_sgd.predict(X_val)

    acc, prec, rec = eval_metrics(y_val, y_pred)

    mlflow.log_param("model_type", "SGDClassifier")
    mlflow.log_param("alpha", best_sgd.alpha)
    mlflow.log_param("loss", best_sgd.loss)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)

    signature = infer_signature(X_train, best_sgd.predict(X_train))
    mlflow.sklearn.log_model(best_sgd, "model", signature=signature)
    print(f"SGD Classifier Finished. Accuracy: {acc:.4f}")


params_rf = {"n_estimators": 100, "max_depth": 6}

with mlflow.start_run(run_name="Random_Forest_Classifier"):
    rf = RandomForestClassifier(
        n_estimators=params_rf["n_estimators"],
        max_depth=params_rf["max_depth"],
        random_state=42,
    )
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_val)
    acc_rf, prec_rf, rec_rf = eval_metrics(y_val, y_pred_rf)

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", params_rf["n_estimators"])
    mlflow.log_param("max_depth", params_rf["max_depth"])
    mlflow.log_metric("accuracy", acc_rf)
    mlflow.log_metric("precision", prec_rf)
    mlflow.log_metric("recall", rec_rf)

    signature_rf = infer_signature(X_train, rf.predict(X_train))
    mlflow.sklearn.log_model(rf, "model", signature=signature_rf)
    print(f"RF Classifier Finished. Accuracy: {acc_rf:.4f}")
