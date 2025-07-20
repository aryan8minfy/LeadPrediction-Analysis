import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def safe_log1p(x):
    """
    Safely apply log1p transformation, handling negative and NaN values
    """
    # Convert to numpy array if not already
    x_array = np.asarray(x)
    
    # Create a copy to avoid modifying the original
    result = x_array.copy()
    
    # Handle NaN values
    nan_mask = np.isnan(result)
    if nan_mask.any():
        # Replace NaN with 0
        result[nan_mask] = 0
    
    # Handle negative values
    negative_mask = result < 0
    if negative_mask.any():
        # Replace negative values with 0
        result[negative_mask] = 0
    
    # Apply log1p
    result = np.log1p(result)
    
    # Final check for any invalid values
    invalid_mask = ~np.isfinite(result)
    if invalid_mask.any():
        result[invalid_mask] = 0
    
    return result

class BinaryMapper(BaseEstimator, TransformerMixin):
    """
    Maps 'Yes'/'No' and similar binary values to 1/0
    """
    def __init__(self):
        self.mapping = {
            'yes': 1, 'no': 0,
            'Yes': 1, 'No': 0,
            'YES': 1, 'NO': 0,
            'y': 1, 'n': 0,
            'Y': 1, 'N': 0,
            'true': 1, 'false': 0,
            'True': 1, 'False': 0,
            'TRUE': 1, 'FALSE': 0,
            '1': 1, '0': 0,
            1: 1, 0: 0,
            1.0: 1, 0.0: 0
        }
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        Transform binary categorical values to numeric
        """
        X_transformed = X.copy()
        
        for col in X_transformed.columns:
            # Check if column contains binary-like values
            unique_vals = X_transformed[col].dropna().unique()
            
            # If column looks binary, apply mapping
            if len(unique_vals) <= 3:  # Allow for NaN as third value
                mapped_values = X_transformed[col].map(self.mapping)
                
                # If mapping was successful for most values, use it
                if mapped_values.notna().sum() > len(X_transformed) * 0.5:
                    X_transformed[col] = mapped_values
                    
                    # Fill any remaining NaN values with 0
                    X_transformed[col] = X_transformed[col].fillna(0)
                else:
                    # If not binary, fill NaN with most frequent value
                    if X_transformed[col].notna().any():
                        mode_val = X_transformed[col].mode()[0] if len(X_transformed[col].mode()) > 0 else 0
                        X_transformed[col] = X_transformed[col].fillna(mode_val)
                    else:
                        X_transformed[col] = 0
        
        return X_transformed
