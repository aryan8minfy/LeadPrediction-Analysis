import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from preprocessing_pipeline import build_preprocessing_pipeline

# -------------------------------------------
# Step 1: Load Data and Preprocessing
# -------------------------------------------
def load_data(data_path):
    df = pd.read_csv(data_path)
    preprocessor, X, y = build_preprocessing_pipeline(df)
    return X, y, preprocessor

# -------------------------------------------
# Step 2: Load Saved Model
# -------------------------------------------
def load_model(model_path):
    return joblib.load(model_path)

# -------------------------------------------
# Step 3: Plot Confusion Matrix
# -------------------------------------------
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

# -------------------------------------------
# Step 4: Plot ROC Curve
# -------------------------------------------
def plot_roc_curve(y_true, y_scores, title="ROC Curve"):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.show()

# -------------------------------------------
# Step 5: Extract Feature Names Safely
# -------------------------------------------
def get_column_names_from_column_transformer(column_transformer):
    column_names = []
    for name, transformer, columns in column_transformer.transformers_:
        if transformer == 'drop':
            continue
        elif transformer == 'passthrough':
            column_names.extend(columns)
        else:
            try:
                if hasattr(transformer, 'get_feature_names_out'):
                    names = transformer.get_feature_names_out()
                elif hasattr(transformer, 'named_steps'):
                    last_step = list(transformer.named_steps.values())[-1]
                    if hasattr(last_step, 'get_feature_names_out'):
                        names = last_step.get_feature_names_out()
                    else:
                        names = [f"{name}_{col}" for col in columns]
                else:
                    names = [f"{name}_{col}" for col in columns]
                column_names.extend(names)
            except Exception:
                column_names.extend([f"{name}_{col}" for col in columns])
    return column_names

# -------------------------------------------
# Step 6: Plot Feature Importance
# -------------------------------------------
def plot_feature_importance(model_pipeline):
    try:
        model = model_pipeline.named_steps["model"]
        preprocessor = model_pipeline.named_steps["preprocessing"]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = get_column_names_from_column_transformer(preprocessor)

            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)[:20]

            plt.figure(figsize=(10, 6))
            sns.barplot(x=feat_imp.values, y=feat_imp.index)
            plt.title("Top 20 Feature Importances")
            plt.xlabel("Importance Score")
            plt.tight_layout()
            plt.show()
        else:
            print("[INFO] Model does not support feature importances.")
    except Exception as e:
        print(f"[INFO] Feature importance not available: {e}")

# -------------------------------------------
# Step 7: Main Evaluation
# -------------------------------------------
def evaluate():
    data_path = "Lead Scoring.csv"
    model_path = "models/best_model_XGBClassifier.pkl"

    # Load preprocessed data
    X, y, _ = load_data(data_path)
    _, X_test, _, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Load trained model pipeline
    model = load_model(model_path)

    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Print evaluation metrics
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

    # Plot visuals
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)
    plot_feature_importance(model)

if __name__ == "__main__":
    evaluate()
