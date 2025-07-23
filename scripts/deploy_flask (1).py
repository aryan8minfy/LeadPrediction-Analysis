import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
import logging

# ------------------------------
# Logging Setup
# ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# Load Best Model
# ------------------------------
MODEL_PATH = "/usr/local/airflow/tmp/best_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ No trained model found. Expected at /usr/local/airflow/tmp/best_model.pkl")

logger.info(f"📦 Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
logger.info("✅ Model loaded successfully")

# ------------------------------
# Flask App
# ------------------------------
app = Flask(__name__, template_folder='/usr/local/airflow/tmp')

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            return "❌ No file selected", 400

        file = request.files["file"]
        try:
            df = pd.read_csv(file)
            preds = model.predict(df)
            df["prediction"] = preds

            logger.info("✅ Prediction completed")

            table_html = df.to_html(classes="table table-bordered", index=False, justify="center")
            return render_template("upload.html", table=table_html)

        except Exception as e:
            logger.error(f"❌ Error in prediction: {e}")
            return f"❌ Error: {e}", 500

    return render_template("upload.html")

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        df = pd.read_csv(file)
        preds = model.predict(df)
        df["prediction"] = preds
        return df.to_json(orient="records")
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

# ------------------------------
# Start App
# ------------------------------
if __name__ == "__main__":
    logger.info("🚀 Starting Flask app...")
    app.run(host="0.0.0.0", port=8050)

