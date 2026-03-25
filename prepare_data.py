# prepare_data.py
from sklearn.datasets import load_iris
import pandas as pd
import os

def main():
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Load iris dataset
    iris = load_iris()
    
    # Create DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Save to CSV
    df.to_csv("data/iris.csv", index=False)
    print("Dataset saved to data/iris.csv")

if __name__ == "__main__":
    main()