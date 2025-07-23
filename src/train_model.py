import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import warnings
warnings.filterwarnings("ignore")

from preprocessing_pipeline import build_preprocessing_pipeline

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    results = []
    best_f1 = 0
    best_model = None
    best_name = ""
    best_run_id = None

    models = {
        "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "RandomForestClassifier": RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=10, random_state=42),
        "XGBClassifier": XGBClassifier(scale_pos_weight=1.6, use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    mlflow.set_experiment("Sales_Conversion_Prediction")

    for name, model in models.items():
        print(f"\nTraining {name}...")
        with mlflow.start_run(run_name=name) as run:
            run_id = run.info.run_id

            pipeline = Pipeline([
                ('preprocessing', preprocessor),
                ('model', model)
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            print(f"{name} - Accuracy: {acc:.4f} | F1: {f1:.4f}")
            print(classification_report(y_test, y_pred))

            mlflow.log_param("model_name", name)
            mlflow.log_metrics({"accuracy": acc, "f1_score": f1})

            # Log model
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            
            results.append((name, f1, acc, pipeline, run_id))

            # Track the best model
            if f1 > best_f1:
                best_f1 = f1
                best_model = pipeline
                best_name = name
                best_run_id = run_id

    return best_name, best_model, results, best_run_id

def register_best_model(best_run_id, model_name="Sales_Conversion_Model"):
    from mlflow.exceptions import MlflowException
    client = MlflowClient()
    
    # Create the registered model if not already there
    try:
        client.create_registered_model(model_name)
        print(f"✅ Created new registered model: {model_name}")
    except MlflowException as e:
        if "already exists" in str(e):
            print(f"[INFO] Registered model '{model_name}' already exists.")
        else:
            raise

    # Register new version
    model_uri = f"runs:/{best_run_id}/model"

    mv = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=best_run_id
    )
    print(f"✅ Registered model version {mv.version}")

    # Transition to staging
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Staging"
    )
    print(f"🚀 Transitioned model to 'Staging'")


def train_pipeline(data_path="Lead Scoring.csv"):
    # Load data
    df = pd.read_csv(data_path)

    # Build pipeline
    global preprocessor
    preprocessor, X, y = build_preprocessing_pipeline(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Train and evaluate
    best_name, best_model, all_results, best_run_id = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Save best model locally
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, f"models/best_model_{best_name}.pkl")
    print(f"\nBest model: {best_name} with F1-score: {max(r[1] for r in all_results):.4f}")
    print(f"Model saved to models/best_model_{best_name}.pkl")

    # Register the best model in MLflow Registry
    register_best_model(best_run_id)


if __name__ == "__main__":
    train_pipeline()
