import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# -----------------------------------------
# 📐 Safe Log Transformation
# -----------------------------------------
def safe_log1p(x):
    """Applies np.log1p but replaces -inf/inf/nan with 0."""
    try:
        x = np.array(x, dtype=np.float64)
        x = np.log1p(np.abs(x))
        x[np.isinf(x)] = 0
        x[np.isnan(x)] = 0
        return x
    except Exception as e:
        print(f"❌ Log transform failed: {e}")
        return x

# -----------------------------------------
# ✅ BinaryMapper for Yes/No/Unknown to 1/0
# -----------------------------------------
class BinaryMapper(BaseEstimator, TransformerMixin):
    """
    Maps 'Yes' to 1, 'No' or 'Unknown' to 0.
    Used for binary categorical fields in lead scoring dataset.
    """
    def __init__(self):
        self.mapping = {"Yes": 1, "No": 0, "Unknown": 0}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.replace(self.mapping).infer_objects()
        elif isinstance(X, pd.Series):
            return X.replace(self.mapping).infer_objects()
        else:
            raise ValueError("BinaryMapper expects pandas DataFrame or Series")
