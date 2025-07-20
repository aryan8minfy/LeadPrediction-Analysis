
import pandas as pd
import argparse
import boto3
from scipy.stats import ks_2samp
import numpy as np

def read_csv_from_s3(s3_key, bucket='july17storagebucket'):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    return pd.read_csv(obj['Body'])

def detect_drift(base_df, new_df, numeric_columns, threshold=0.05):
    drifted_features = []
    for col in numeric_columns:
        if col in base_df.columns and col in new_df.columns:
            stat, p_value = ks_2samp(base_df[col].dropna(), new_df[col].dropna())
            if p_value < threshold:
                drifted_features.append(col)
    return drifted_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3_key', required=True, help='S3 key of new data file')
    args = parser.parse_args()

    baseline_key = 'data/Lead Scoring.csv'
    bucket = 'july17storagebucket'

    print("📥 Reading baseline and new data from S3...")
    base_df = read_csv_from_s3(baseline_key, bucket)
    new_df = read_csv_from_s3(args.s3_key, bucket)

    numeric_columns = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']
    drifted = detect_drift(base_df, new_df, numeric_columns)

    if drifted:
        print(f"🚨 Data drift detected in features: {drifted}")
        print("drift")  # signal to DAG
    else:
        print("✅ No significant drift detected.")

if __name__ == "__main__":
    main()
