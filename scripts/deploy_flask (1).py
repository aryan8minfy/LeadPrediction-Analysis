
import joblib
import os
import shutil

def main():
    source_path = None

    # Find best model in temp folder
    for fname in os.listdir("/usr/local/airflow/tmp"):
        if fname.startswith("best_model_") and fname.endswith(".pkl"):
            source_path = os.path.join("/usr/local/airflow/tmp", fname)
            break

    if not source_path:
        raise FileNotFoundError("❌ No trained model file found in /usr/local/airflow/tmp/")

    deploy_path = "/usr/local/airflow/serving/model.pkl"
    os.makedirs(os.path.dirname(deploy_path), exist_ok=True)

    shutil.copyfile(source_path, deploy_path)
    print(f"✅ Model deployed to: {deploy_path}")

if __name__ == "__main__":
    main()
