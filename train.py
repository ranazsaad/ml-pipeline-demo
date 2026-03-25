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

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def main():
    # Set MLflow tracking URI (use local server)
    mlflow.set_tracking_uri("http://localhost:5001")
    
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
    
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Log to MLflow
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("test_size", 0.2)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        # Get the run ID
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        
        # Create model_info.txt with the run ID
        with open("model_info.txt", "w") as f:
            f.write(run_id)
        
        print("✅ model_info.txt created successfully")
        print(f"📊 View run at: http://localhost:5001/#/experiments/0/runs/{run_id}")

if __name__ == "__main__":
    main()