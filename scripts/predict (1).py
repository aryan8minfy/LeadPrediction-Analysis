import os
import joblib
import pandas as pd

def main():
    tmp_dir = "/usr/local/airflow/tmp"
    model_file = None
    
    # Dynamically find best_model_*.pkl
    for fname in os.listdir(tmp_dir):
        if fname.startswith("best_model_") and fname.endswith(".pkl"):
            model_file = os.path.join(tmp_dir, fname)
            break

    if not model_file:
        raise FileNotFoundError("❌ No best_model_*.pkl found in /usr/local/airflow/tmp/")

    print(f"✅ Using model: {model_file}")
    
    model = joblib.load(model_file)

    # Dummy prediction (replace with real test set)
    dummy_input = pd.read_csv(f"{tmp_dir}/X.csv").head(5)
    preds = model.predict(dummy_input)
    
    print("🤖 Predictions:", preds)

if __name__ == "__main__":
    main()
