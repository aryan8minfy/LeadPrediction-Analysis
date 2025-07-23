import argparse
import boto3
import pandas as pd
import numpy as np
import os
import io
import joblib
import logging

# ------------------------
# Logging setup
# ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------
# Run Prediction
# ------------------------
def run_prediction(s3_key):
    bucket = "storage21julybucket"
    logger.info(f"📥 Loading new data from s3://{bucket}/{s3_key}")
    
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=s3_key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        logger.info(f"✅ New data loaded: {df.shape}")
    except Exception as e:
        logger.error(f"❌ Failed to load data from S3: {e}")
        raise

    # ✅ Load model from fixed path
    model_path = "/usr/local/airflow/tmp/best_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ No trained model found. best_model.pkl is missing.")

    try:
        model = joblib.load(model_path)
        logger.info(f"✅ Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

    # ✅ Predict
    try:
        predictions = model.predict(df)
        df["prediction"] = predictions
        logger.info(f"✅ Predictions generated. Sample:\n{df[['prediction']].head()}")
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise

    # ✅ Save output back to S3
    try:
        output_buffer = io.StringIO()
        df.to_csv(output_buffer, index=False)
        result_key = f"tmp/predictions_{os.path.basename(s3_key)}"
        s3.put_object(Bucket=bucket, Key=result_key, Body=output_buffer.getvalue())
        logger.info(f"📤 Predictions saved to s3://{bucket}/{result_key}")
    except Exception as e:
        logger.error(f"❌ Failed to upload predictions: {e}")
        raise

# ------------------------
# Entrypoint
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3_key", required=True, help="S3 key to CSV file in new_data/")
    args = parser.parse_args()
    run_prediction(args.s3_key)
