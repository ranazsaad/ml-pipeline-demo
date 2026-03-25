# train.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import random
import json

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def main():
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
    
    # Create model_info.txt with run ID
    run_id = "model_" + str(random.randint(10000, 99999))
    with open("model_info.txt", "w") as f:
        f.write(run_id)
    
    # Save accuracy to a file
    with open("model_accuracy.txt", "w") as f:
        f.write(str(accuracy))
    
    # Save both in JSON format
    model_data = {
        "run_id": run_id,
        "accuracy": accuracy,
        "model_type": "random_forest",
        "n_estimators": 100
    }
    with open("model_data.json", "w") as f:
        json.dump(model_data, f)
    
    print("✅ model_info.txt created successfully")
    print(f"✅ model_accuracy.txt created with accuracy: {accuracy}")
    print(f"✅ model_data.json created")

if __name__ == "__main__":
    main()