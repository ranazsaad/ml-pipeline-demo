# train_low_accuracy.py
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def main():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5001")
    
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=42
    )
    
    print("Training Decision Tree model (will have lower accuracy)...")
    # Use a shallow tree for lower accuracy
    model = DecisionTreeClassifier(max_depth=1, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "decision_tree")
        mlflow.log_param("max_depth", 2)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
        
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        
        with open("model_info.txt", "w") as f:
            f.write(run_id)
        
        print(f"📊 View run at: http://localhost:5001/#/experiments/0/runs/{run_id}")

if __name__ == "__main__":
    main()