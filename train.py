# train.py
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import random
import os
import json

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def main():
    # Use environment variable for MLflow tracking URI (optional)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"🔗 MLflow Tracking URI: {tracking_uri}")
    
    print("Loading Iris dataset...")
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print(f"Dataset shape: {X.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training Random Forest model...")
    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions and calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Model Accuracy: {accuracy:.4f}")
    
    # Log to MLflow if server is available
    try:
        with mlflow.start_run() as run:
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "model")
            run_id = run.info.run_id
            print(f"📝 MLflow Run ID: {run_id}")
    except Exception as e:
        print(f"⚠️ MLflow logging failed: {e}")
        run_id = "local-run"
    
    # Create model_info.txt with run ID and accuracy
    with open("model_info.txt", "w") as f:
        f.write(run_id)
    
    # Also save accuracy to a separate file for easy access
    with open("model_accuracy.txt", "w") as f:
        f.write(str(accuracy))
    
    # Save both in JSON format for reliability
    model_data = {
        "run_id": run_id,
        "accuracy": accuracy,
        "model_type": "random_forest"
    }
    with open("model_data.json", "w") as f:
        json.dump(model_data, f)
    
    print("✅ model_info.txt created successfully")
    print(f"✅ model_accuracy.txt created with accuracy: {accuracy}")
    print(f"✅ model_data.json created with all data")

if __name__ == "__main__":
    main()