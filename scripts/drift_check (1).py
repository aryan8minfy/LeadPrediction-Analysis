import argparse
import boto3
import pandas as pd
import io
import os
import logging
import numpy as np
from scipy.stats import ks_2samp
from utils import safe_log1p, BinaryMapper

# --------------------------------
# Logging setup
# --------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------
# Read CSV from S3
# --------------------------------
def read_csv_from_s3(bucket, key):
    logger.info(f"📥 Reading from s3://{bucket}/{key}")
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

# --------------------------------
# Detect Drift Function
# --------------------------------
def detect_drift(reference_df, new_df, threshold=0.1):
    drift_report = {}
    drift_detected = False

    common_cols = list(set(reference_df.columns) & set(new_df.columns))
    logger.info(f"🔍 Comparing {len(common_cols)} common columns...")

    for col in common_cols:
        if pd.api.types.is_numeric_dtype(reference_df[col]) and pd.api.types.is_numeric_dtype(new_df[col]):
            ref_col = reference_df[col].dropna()
            new_col = new_df[col].dropna()

            if len(ref_col) > 0 and len(new_col) > 0:
                stat, p_value = ks_2samp(ref_col, new_col)
                drift = p_value < threshold

                drift_report[col] = {
                    "p_value": round(p_value, 4),
                    "drift_detected": drift
                }

                if drift:
                    drift_detected = True

    return drift_report, drift_detected

# --------------------------------
# Main Logic
# --------------------------------
def main(s3_key, ref_key):
    bucket = "storage21julybucket"

    try:
        reference_df = read_csv_from_s3(bucket, ref_key)
        new_df = read_csv_from_s3(bucket, s3_key)
    except Exception as e:
        logger.error(f"❌ Failed to read data: {e}")
        raise

    try:
        logger.info(f"✅ Data loaded. Reference: {reference_df.shape}, New: {new_df.shape}")
        drift_report, drift_detected = detect_drift(reference_df, new_df)

        logger.info("📊 Drift Detection Report:")
        for col, result in drift_report.items():
            logger.info(f"   - {col}: drift={result['drift_detected']} (p={result['p_value']})")

        # Save result to /tmp for Airflow to branch
        os.makedirs("/tmp", exist_ok=True)
        result_text = "drift" if drift_detected else "no_drift"
        with open("/tmp/drift_result.txt", "w") as f:
            f.write(result_text)

        if drift_detected:
            print("🚨 Drift Detected! Model retraining is recommended.")
        else:
            print("✅ No significant drift detected. Proceeding with current model.")

    except Exception as e:
        logger.error(f"❌ Drift detection error: {e}")
        raise

# --------------------------------
# Entry Point
# --------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3_key", required=True, help="S3 key for new data (in new_data/)")
    parser.add_argument("--ref_key", required=True, help="S3 key for reference data (e.g., data/train.csv)")
    args = parser.parse_args()
    main(args.s3_key, args.ref_key)

