# check_threshold.py
import sys
import os
import json

def main():
    print("🔍 Checking model accuracy threshold...")
    
    # Try to read accuracy from files
    accuracy = None
    
    # First try model_accuracy.txt
    if os.path.exists("model_accuracy.txt"):
        try:
            with open("model_accuracy.txt", "r") as f:
                accuracy = float(f.read().strip())
            print(f"📊 Read accuracy from model_accuracy.txt: {accuracy:.4f}")
        except Exception as e:
            print(f"⚠️ Error reading model_accuracy.txt: {e}")
    
    # If not found, try model_data.json
    if accuracy is None and os.path.exists("model_data.json"):
        try:
            with open("model_data.json", "r") as f:
                model_data = json.load(f)
                accuracy = model_data.get("accuracy")
                run_id = model_data.get("run_id")
                print(f"📊 Read from model_data.json - Run ID: {run_id}, Accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"⚠️ Error reading model_data.json: {e}")
    
    # If still not found, try model_info.txt (but that only has run_id, not accuracy)
    if accuracy is None and os.path.exists("model_info.txt"):
        try:
            with open("model_info.txt", "r") as f:
                run_id = f.read().strip()
            print(f"📊 Found model_info.txt with Run ID: {run_id}")
            print("⚠️ But no accuracy file found!")
        except Exception as e:
            print(f"⚠️ Error reading model_info.txt: {e}")
    
    if accuracy is None:
        print("❌ No accuracy file found (model_accuracy.txt or model_data.json)")
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