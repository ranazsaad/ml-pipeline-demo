# check_threshold.py
import sys
import os
import json

def main():
    print("🔍 Checking model accuracy threshold...")
    
    # Check if model_accuracy.txt exists (simplest approach)
    if os.path.exists("model_accuracy.txt"):
        try:
            with open("model_accuracy.txt", "r") as f:
                accuracy = float(f.read().strip())
            print(f"📊 Read accuracy from model_accuracy.txt: {accuracy:.4f}")
        except Exception as e:
            print(f"❌ Error reading model_accuracy.txt: {e}")
            sys.exit(1)
    
    # Alternative: Check model_data.json
    elif os.path.exists("model_data.json"):
        try:
            with open("model_data.json", "r") as f:
                model_data = json.load(f)
                accuracy = model_data.get("accuracy")
                run_id = model_data.get("run_id")
                print(f"📊 Read from model_data.json - Run ID: {run_id}, Accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"❌ Error reading model_data.json: {e}")
            sys.exit(1)
    
    # Fallback: Read from model_info.txt and try MLflow
    elif os.path.exists("model_info.txt"):
        try:
            with open("model_info.txt", "r") as f:
                run_id = f.read().strip()
            print(f"📊 Found model_info.txt with Run ID: {run_id}")
            
            # Try to get accuracy from MLflow
            try:
                import mlflow
                import os
                tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
                mlflow.set_tracking_uri(tracking_uri)
                run = mlflow.get_run(run_id)
                accuracy = run.data.metrics.get("accuracy")
                print(f"📊 Got accuracy from MLflow: {accuracy:.4f}")
            except Exception as e:
                print(f"⚠️ Could not get accuracy from MLflow: {e}")
                print("❌ Cannot determine accuracy")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading files: {e}")
            sys.exit(1)
    
    else:
        print("❌ No accuracy files found (model_accuracy.txt, model_data.json, or model_info.txt)")
        print("Current directory contents:")
        for file in os.listdir("."):
            print(f"  - {file}")
        sys.exit(1)
    
    # Check threshold
    threshold = 0.85
    print(f"🎯 Model Accuracy: {accuracy:.4f}")
    print(f"📏 Threshold: {threshold}")
    
    if accuracy >= threshold:
        print(f"✅ Accuracy ({accuracy:.4f}) meets threshold ({threshold})")
        if 'run_id' in locals():
            print(f"🐳 Building Docker image for Run ID: {run_id}")
        else:
            print("🐳 Building Docker image for model")
        sys.exit(0)
    else:
        print(f"❌ Accuracy ({accuracy:.4f}) is below threshold ({threshold})")
        sys.exit(1)

if __name__ == "__main__":
    main()