# check_threshold.py
import mlflow
import sys
import os

def main():
    # Get MLflow tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"🔗 MLflow Tracking URI: {tracking_uri}")
    
    # Check if model_info.txt exists
    if not os.path.exists("model_info.txt"):
        print("❌ model_info.txt not found in current directory")
        print("Current directory contents:")
        for file in os.listdir("."):
            print(f"  - {file}")
        sys.exit(1)
    
    # Read run ID
    try:
        with open("model_info.txt", "r") as f:
            run_id = f.read().strip()
        if not run_id:
            print("❌ model_info.txt is empty")
            sys.exit(1)
        print(f"📊 Checking model with Run ID: {run_id}")
    except Exception as e:
        print(f"❌ Error reading model_info.txt: {e}")
        sys.exit(1)
    
    # List all runs to debug
    try:
        print("Listing all MLflow runs...")
        experiment = mlflow.get_experiment_by_name("Default")
        if experiment:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            print(f"Found {len(runs)} runs")
            if len(runs) > 0:
                print("Available runs:")
                for idx, row in runs.iterrows():
                    print(f"  - {row['run_id']}: accuracy={row.get('metrics.accuracy', 'N/A')}")
    except Exception as e:
        print(f"Warning: Could not list runs: {e}")
    
    # Get the specific run
    try:
        run = mlflow.get_run(run_id)
        print(f"✅ Found run: {run.info.run_id}")
        
        # Extract accuracy metric
        accuracy = run.data.metrics.get("accuracy")
        
        if accuracy is None:
            print("❌ Error: Accuracy metric not found in the run")
            print(f"Available metrics: {list(run.data.metrics.keys())}")
            sys.exit(1)
        
        print(f"🎯 Model Accuracy: {accuracy:.4f}")
        
        # Check threshold
        threshold = 0.85
        if accuracy >= threshold:
            print(f"✅ Accuracy ({accuracy:.4f}) meets threshold ({threshold})")
            print(f"🐳 Building Docker image for Run ID: {run_id}")
            sys.exit(0)
        else:
            print(f"❌ Accuracy ({accuracy:.4f}) is below threshold ({threshold})")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error fetching run from MLflow: {e}")
        print(f"Run ID attempted: {run_id}")
        sys.exit(1)

if __name__ == "__main__":
    main()