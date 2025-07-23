import argparse
import boto3
import pandas as pd
import numpy as np
import io
import os
import logging
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

# Import shared utilities
from utils import safe_log1p, BinaryMapper

# --------------------------------
# Logging setup
# --------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

log_transformer = FunctionTransformer(safe_log1p, validate=False)

# --------------------------------
def remove_outliers_iqr(df, columns, threshold=1.5):
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

def simplify_categories(df, cat_cols, threshold=0.01):
    for col in cat_cols:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            rare_labels = freq[freq < threshold].index
            df[col] = df[col].apply(lambda x: 'Other' if x in rare_labels else x)
    return df

def clean_data(df):
    logger.info("🧹 Starting enhanced data cleaning...")
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna('Unknown')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
        df[col] = df[col].replace([np.inf, -np.inf], df[col].median())
    logger.info("✅ Data cleaning completed")
    return df

def build_preprocessing_pipeline(df):
    logger.info("🔨 Building preprocessing pipeline...")
    drop_columns = [
        'Prospect ID', 'Lead Number', 'Tags', 'Lead Quality', 'Lead Profile',
        'How did you hear about X Education', 'What matters most to you in choosing a course',
        'Asymmetrique Activity Index', 'Asymmetrique Profile Index',
        'Asymmetrique Activity Score', 'Asymmetrique Profile Score'
    ]
    df.drop(columns=drop_columns, errors='ignore', inplace=True)
    df.drop_duplicates(inplace=True)
    df = clean_data(df)

    categorical_candidates = ['Lead Source', 'Last Activity', 'Specialization', 'City']
    df = simplify_categories(df, categorical_candidates)
    df = remove_outliers_iqr(df, ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit'])

    if 'Converted' not in df.columns:
        raise ValueError("❌ 'Converted' column missing")

    y = df['Converted']
    X = df.drop(columns='Converted')

    numeric_features = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']
    binary_features = [
        'Do Not Email', 'Do Not Call', 'Search', 'Magazine', 'Newspaper Article',
        'X Education Forums', 'Newspaper', 'Digital Advertisement',
        'Through Recommendations', 'Receive More Updates About Our Courses',
        'Update me on Supply Chain Content', 'Get updates on DM Content',
        'I agree to pay the amount through cheque', 'A free copy of Mastering The Interview'
    ]
    categorical_features = [
        'Lead Origin', 'Lead Source', 'Last Activity', 'Country',
        'Specialization', 'What is your current occupation',
        'City', 'Last Notable Activity'
    ]

    numeric_features = [c for c in numeric_features if c in X.columns]
    binary_features = [c for c in binary_features if c in X.columns]
    categorical_features = [c for c in categorical_features if c in X.columns]

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log', log_transformer),
        ('scaler', StandardScaler())
    ])

    binary_pipeline = Pipeline([
        ('mapper', BinaryMapper()),
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('bin', binary_pipeline, binary_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    return preprocessor, X, y

# --------------------------------
def main(s3_key, bucket):
    logger.info(f"🔍 Reading s3://{bucket}/{s3_key}")
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=s3_key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        logger.info("✅ Data read successfully")

        preprocessor, X, y = build_preprocessing_pipeline(df)
        os.makedirs("/usr/local/airflow/tmp", exist_ok=True)
        X.to_csv("/usr/local/airflow/tmp/X.csv", index=False)
        y.to_csv("/usr/local/airflow/tmp/y.csv", index=False)
        joblib.dump(preprocessor, "/usr/local/airflow/tmp/preprocessor.pkl")
        logger.info("📁 Files saved to /usr/local/airflow/tmp")

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise

# --------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3_key", required=True, help="Key of new CSV inside new_data/")
    parser.add_argument("--bucket", default="storage21julybucket", help="S3 bucket name")
    args = parser.parse_args()
    main(args.s3_key, args.bucket)
