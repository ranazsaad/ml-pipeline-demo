# train_low_accuracy.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import random
import json

def main():
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print("Training Decision Tree model (will have lower accuracy)...")
    # Use a shallow tree for lower accuracy
    model = DecisionTreeClassifier(max_depth=2, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Model Accuracy: {accuracy:.4f}")
    
    # Create output files
    run_id = "model_" + str(random.randint(10000, 99999))
    
    with open("model_info.txt", "w") as f:
        f.write(run_id)
    
    with open("model_accuracy.txt", "w") as f:
        f.write(str(accuracy))
    
    model_data = {
        "run_id": run_id,
        "accuracy": accuracy,
        "model_type": "decision_tree",
        "max_depth": 2
    }
    with open("model_data.json", "w") as f:
        json.dump(model_data, f)
    
    print("✅ Files created successfully")

if __name__ == "__main__":
    main()