import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import json

# Load dataset
df = pd.read_csv("data/data.csv")

print("Columns:", df.columns)

# Target column
TARGET = "target"

X = df.drop(TARGET, axis=1)
y = df[TARGET]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# MLflow setup
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("2022BCS0209_experiment")

with mlflow.start_run():

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, artifact_path="model")

    joblib.dump(model, "model.joblib")

    with open("metrics.json", "w") as f:
        json.dump({
            "accuracy": acc,
            "f1": f1,
            "name": "Harshitha",
            "rollno": "2022BCS0209"
        }, f)

print("Training Completed!")