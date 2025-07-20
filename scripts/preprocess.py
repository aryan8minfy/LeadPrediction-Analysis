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

# --------------------------------
# Create function transformer using shared function
# --------------------------------
log_transformer = FunctionTransformer(safe_log1p, validate=False)

# --------------------------------
# Remove outliers using IQR
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

# --------------------------------
# Replace rare categories with "Other"
# --------------------------------
def simplify_categories(df, cat_cols, threshold=0.01):
    for col in cat_cols:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            rare_labels = freq[freq < threshold].index
            df[col] = df[col].apply(lambda x: 'Other' if x in rare_labels else x)
    return df

# --------------------------------
# Enhanced data cleaning
# --------------------------------
def clean_data(df):
    """Enhanced data cleaning with better NaN handling"""
    logger.info("🧹 Starting enhanced data cleaning...")

    # Fill missing values in object columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna('Unknown')
        logger.info(f"   Filled NaN in {col} with 'Unknown'")

    # Handle numeric columns more carefully
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"   Filled {df[col].isnull().sum()} NaN values in {col} with median: {median_val}")

        # Replace infinite values
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            df[col] = df[col].replace([np.inf, -np.inf], df[col].median())
            logger.info(f"   Replaced {inf_count} infinite values in {col}")

    logger.info("✅ Data cleaning completed")
    return df

# --------------------------------
# Main preprocessing logic
# --------------------------------
def build_preprocessing_pipeline(df):
    logger.info("🔨 Building preprocessing pipeline...")

    # Drop unnecessary columns
    drop_columns = [
        'Prospect ID', 'Lead Number', 'Tags', 'Lead Quality', 'Lead Profile',
        'How did you hear about X Education', 'What matters most to you in choosing a course',
        'Asymmetrique Activity Index', 'Asymmetrique Profile Index',
        'Asymmetrique Activity Score', 'Asymmetrique Profile Score'
    ]

    initial_shape = df.shape
    df.drop(columns=drop_columns, errors='ignore', inplace=True)
    logger.info(f"   Dropped columns: {initial_shape[1] - df.shape[1]} columns removed")

    # Remove duplicates
    dup_count = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    logger.info(f"   Removed {dup_count} duplicate rows")

    # Enhanced data cleaning
    df = clean_data(df)

    # Simplify categorical variables
    categorical_candidates = ['Lead Source', 'Last Activity', 'Specialization', 'City']
    df = simplify_categories(df, categorical_candidates)
    logger.info("   Simplified categorical variables")

    # Remove outliers
    outlier_cols = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']
    pre_outlier_shape = df.shape
    df = remove_outliers_iqr(df, outlier_cols)
    logger.info(f"   Outlier removal: {pre_outlier_shape[0] - df.shape[0]} rows removed")

    # Validate target column
    if 'Converted' not in df.columns:
        raise ValueError("'Converted' column is missing from input data")

    y = df['Converted']
    X = df.drop(columns='Converted')

    # Define feature groups with validation
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

    # Filter features that actually exist in the data
    numeric_features = [col for col in numeric_features if col in X.columns]
    binary_features = [col for col in binary_features if col in X.columns]
    categorical_features = [col for col in categorical_features if col in X.columns]

    logger.info(f"   Feature counts - Numeric: {len(numeric_features)}, Binary: {len(binary_features)}, Categorical: {len(categorical_features)}")

    # Enhanced preprocessing pipelines
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
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))  # ✅ change made here
    ])

    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('bin', binary_pipeline, binary_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop'
    )

    logger.info("✅ Preprocessing pipeline built successfully")
    return preprocessor, X, y

# --------------------------------
# Entry point
# --------------------------------
def main(s3_key):
    bucket = "july17storagebucket"
    logger.info(f"🔍 Reading s3://{bucket}/{s3_key}")
    s3 = boto3.client("s3")

    try:
        obj = s3.get_object(Bucket=bucket, Key=s3_key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        logger.info("✅ Data read successfully")
        logger.info(f"📊 Initial shape: {df.shape}")

        logger.info(f"📈 Data overview:")
        logger.info(f"   - Total missing values: {df.isnull().sum().sum()}")
        logger.info(f"   - Columns with missing values: {df.isnull().any().sum()}")
        logger.info(f"   - Data types: {df.dtypes.value_counts().to_dict()}")

        preprocessor, X, y = build_preprocessing_pipeline(df)
        logger.info(f"✅ Preprocessing complete | X shape: {X.shape} | y shape: {y.shape}")

        logger.info(f"📊 Final data validation:")
        logger.info(f"   - X missing values: {X.isnull().sum().sum()}")
        logger.info(f"   - y missing values: {y.isnull().sum()}")
        logger.info(f"   - Target distribution: {y.value_counts().to_dict()}")

        os.makedirs("/usr/local/airflow/tmp", exist_ok=True)
        X.to_csv("/usr/local/airflow/tmp/X.csv", index=False)
        y.to_csv("/usr/local/airflow/tmp/y.csv", index=False)
        joblib.dump(preprocessor, "/usr/local/airflow/tmp/preprocessor.pkl")
        logger.info("📁 Saved X.csv, y.csv, and preprocessor.pkl to /usr/local/airflow/tmp")

    except Exception as e:
        logger.error(f"❌ Failed to preprocess: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3_key", required=True, help="S3 key to CSV inside 'new_data/' folder")
    args = parser.parse_args()
    main(args.s3_key)
