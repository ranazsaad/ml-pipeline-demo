# check_threshold.py
import mlflow
import sys
import os

def main():
    # Set MLflow tracking URI to local server
    mlflow.set_tracking_uri("http://localhost:5001")
    
    # Read the run ID from the artifact
    try:
        with open("model_info.txt", "r") as f:
            run_id = f.read().strip()
        print(f"📊 Checking model with Run ID: {run_id}")
    except FileNotFoundError:
        print("❌ Error: model_info.txt not found")
        sys.exit(1)
    
    print(f"🔗 Using MLflow tracking URI: http://localhost:5001")
    
    try:
        # Get the run from MLflow
        run = mlflow.get_run(run_id)
        
        # Extract accuracy metric
        accuracy = run.data.metrics.get("accuracy")
        
        if accuracy is None:
            print("❌ Error: Accuracy metric not found in the run")
            sys.exit(1)
        
        print(f"🎯 Model Accuracy: {accuracy:.4f}")
        
        # Check if accuracy meets threshold (0.85)
        threshold = 0.85
        if accuracy >= threshold:
            print(f"✅ Accuracy ({accuracy:.4f}) meets threshold ({threshold})")
            print(f"🐳 Building Docker image for Run ID: {run_id}")
            sys.exit(0)  # Success
        else:
            print(f"❌ Accuracy ({accuracy:.4f}) is below threshold ({threshold})")
            sys.exit(1)  # Fail the pipeline
            
    except Exception as e:
        print(f"❌ Error fetching run from MLflow: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()