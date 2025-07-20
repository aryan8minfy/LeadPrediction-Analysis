import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

from utils import safe_log1p, BinaryMapper

# ---------------------------------------------
# 🚀 Main Training Logic
# ---------------------------------------------
def main():
    print("📂 Loading data...")

    # ✅ Local MLflow availability flag
    mlflow_available = False
    try:
        mlflow.set_tracking_uri("https://mlflow.ap-south-1.studio.ml.amazonaws.com")
        mlflow.set_experiment("Sales_Conversion_Prediction")
        mlflow_available = True
    except Exception as e:
        print("⚠️ MLflow connection failed. Skipping remote logging.")
        print(str(e))

    try:
        X = pd.read_csv('/usr/local/airflow/tmp/X.csv')
        y = pd.read_csv('/usr/local/airflow/tmp/y.csv')
        print("✅ Data loaded successfully")
        print(f"🔍 Validating data after loading...")
        print(f"   X NaN count: {X.isnull().sum().sum()}")
        print(f"   y NaN count: {y.isnull().sum()}")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        raise

    try:
        preprocessor = joblib.load('/usr/local/airflow/tmp/preprocessor.pkl')
        print("✅ Preprocessor loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load preprocessor: {e}")
        raise

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        print(f"✅ Data split: Train {X_train.shape}, Test {X_test.shape}")
    except Exception as e:
        print(f"❌ Split failed: {e}")
        raise

    try:
        print("🔄 Transforming data...")
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Ensure dense numeric data
        if hasattr(X_train_transformed, "toarray"):
            X_train_transformed = X_train_transformed.toarray()
            X_test_transformed = X_test_transformed.toarray()

        X_train_transformed = np.array(X_train_transformed, dtype=np.float64)
        X_test_transformed = np.array(X_test_transformed, dtype=np.float64)

        print(f"✅ Transformed: Train {X_train_transformed.shape}, Test {X_test_transformed.shape}")
        print("🧹 Cleaning transformed data...")
        X_train_transformed = np.nan_to_num(X_train_transformed, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_transformed = np.nan_to_num(X_test_transformed, nan=0.0, posinf=0.0, neginf=0.0)
        print("✅ Transformation clean and complete")
    except Exception as e:
        print(f"❌ Transformation error: {e}")
        raise

    models = {
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        "XGBClassifier": XGBClassifier(
            eval_metric='logloss', scale_pos_weight=1.6, random_state=42, missing=0.0
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42, solver='liblinear'
        )
    }

    best_f1 = 0
    best_model = None
    best_name = ""
    results = {}

    for name, model in models.items():
        print(f"\n🤖 Training {name}...")
        try:
            model.fit(X_train_transformed, y_train.values.ravel())
            print(f"✅ {name} trained")

            y_pred = model.predict(X_test_transformed)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            results[name] = {"accuracy": acc, "f1_score": f1}

            print(f"✅ {name} - Accuracy: {acc:.4f}, F1: {f1:.4f}")
            print(classification_report(y_test, y_pred))

            if mlflow_available:
                try:
                    with mlflow.start_run(run_name=name):
                        mlflow.log_param("model_name", name)
                        mlflow.log_metrics({"accuracy": acc, "f1_score": f1})
                        pipeline = Pipeline([
                            ('preprocessor', preprocessor),
                            ('model', model)
                        ])
                        mlflow.sklearn.log_model(pipeline, artifact_path="model")
                    print(f"✅ MLflow logging done for {name}")
                except Exception as e:
                    print(f"⚠️ MLflow logging failed for {name}: {e}")
                    mlflow_available = False

            if f1 > best_f1:
                best_f1 = f1
                best_model = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])
                best_name = name

        except Exception as e:
            print(f"❌ {name} training failed: {e}")
            import traceback
            traceback.print_exc()

    if best_model:
        try:
            os.makedirs("/usr/local/airflow/tmp", exist_ok=True)
            model_path = f"/usr/local/airflow/tmp/best_model_{best_name}.pkl"
            joblib.dump(best_model, model_path)
            print(f"\n🏆 Best Model: {best_name} (F1: {best_f1:.4f})")
            print(f"📦 Saved to: {model_path}")

            with open("/usr/local/airflow/tmp/training_results.json", "w") as f:
                import json
                json.dump(results, f, indent=2)
            print("📊 Results saved to training_results.json")

        except Exception as e:
            print(f"❌ Failed to save best model: {e}")
            raise
    else:
        raise Exception("❌ No valid models trained!")

if __name__ == "__main__":
    main()
