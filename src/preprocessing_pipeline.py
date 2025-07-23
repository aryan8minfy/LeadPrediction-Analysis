import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


# -----------------------------
# Custom Transformer: BinaryMapper
# -----------------------------
class BinaryMapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mapping = {'Yes': 1, 'No': 0, 'Unknown': 0}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.replace(self.mapping).infer_objects(copy=False)


# -----------------------------
# Log Transformation Function
# -----------------------------
def log_transform(x):
    return np.log1p(x)

log_transformer = FunctionTransformer(log_transform, validate=True)


# -----------------------------
# Outlier Removal using IQR
# -----------------------------
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


# -----------------------------
# Simplify Rare Categories
# -----------------------------
def simplify_categories(df, cat_cols, threshold=0.01):
    for col in cat_cols:
        freq = df[col].value_counts(normalize=True)
        rare_labels = freq[freq < threshold].index
        df[col] = df[col].apply(lambda x: 'Other' if x in rare_labels else x)
    return df


# -----------------------------
# Main Preprocessing Function
# -----------------------------
def build_preprocessing_pipeline(df):
    """
    Preprocessing pipeline with outlier removal, rare category simplification,
    log transformation, scaling, and categorical encoding.
    Returns:
        - preprocessor: sklearn ColumnTransformer
        - X: Features DataFrame
        - y: Target Series
    """

    # Drop irrelevant or high-missing columns
    drop_columns = [
        'Prospect ID', 'Lead Number',
        'Tags', 'Lead Quality', 'Lead Profile',
        'How did you hear about X Education',
        'What matters most to you in choosing a course',
        'Asymmetrique Activity Index', 'Asymmetrique Profile Index',
        'Asymmetrique Activity Score', 'Asymmetrique Profile Score'
    ]
    df = df.drop(columns=drop_columns, errors='ignore')

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Fill missing categorical with 'Unknown'
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna('Unknown')

    # Simplify categories
    categorical_candidates = ['Lead Source', 'Last Activity', 'Specialization', 'City']
    df = simplify_categories(df, categorical_candidates, threshold=0.01)

    # Remove outliers using IQR
    outlier_cols = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']
    df = remove_outliers_iqr(df, outlier_cols)

    # Separate features and target
    y = df['Converted']
    X = df.drop(columns='Converted')

    # Feature groups
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

    # Retain only columns that exist in X
    numeric_features = [col for col in numeric_features if col in X.columns]
    binary_features = [col for col in binary_features if col in X.columns]
    categorical_features = [col for col in categorical_features if col in X.columns]

    # Pipelines
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log', log_transformer),
        ('scaler', StandardScaler())
    ])

    binary_pipeline = Pipeline([
        ('mapper', BinaryMapper())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    # ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('bin', binary_pipeline, binary_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    return preprocessor, X, y
