import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model_path, x_test_path, y_test_path, metrics_path):
    model = joblib.load(model_path)
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).iloc[:, 0]
    
    predictions = model.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    metrics = {
        "Accuracy": float(acc),
        "F1_Score": float(f1)
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    evaluate_model("data/model.joblib", "data/X_test.csv", "data/y_test.csv", "metrics.json")
