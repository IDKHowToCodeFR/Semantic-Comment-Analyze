import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import os
import argparse
from nlp_engine import get_embeddings_batch

def train_intent_model(data_path: str, output_path: str, max_iter: int = 1000):
    print(f"Loading training data from {data_path}...")
    df = pd.read_csv(data_path)
    
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")
    
    print(f"Extracting embeddings for {len(df)} samples (this may take a moment)...")
    X = get_embeddings_batch(df["text"].tolist())
    y = df["label"].tolist()
    
    print("Training Logistic Regression classifier head...")
    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(X, y)
    
    print(f"Saving trained intent model to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(clf, f)
        
    print("Training complete! You can now run the server.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the custom intent classification head.")
    parser.add_argument("--data", default="../../data/training_data.csv", help="Path to training CSV")
    parser.add_argument("--output", default="../../data/local_model_head.pkl", help="Path to save the model")
    
    args = parser.parse_args()
    
    # Resolve absolute paths relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.normpath(os.path.join(base_dir, args.data))
    output_path = os.path.normpath(os.path.join(base_dir, args.output))
    
    train_intent_model(data_path, output_path)
