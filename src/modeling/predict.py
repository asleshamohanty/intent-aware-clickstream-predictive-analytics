import pandas as pd
import pickle
import numpy as np

MODEL_PATH = "models/conversion_model_xgb_enhanced.pkl"

# Load model
with open(MODEL_PATH, "rb") as f:
    clf = pickle.load(f)

# Example function for prediction
def predict_conversion(session_features: pd.DataFrame) -> pd.DataFrame:
    """
    session_features: DataFrame with same columns as training features
    Returns: DataFrame with predicted probability and label
    """
    # Predict probability of conversion
    prob = clf.predict_proba(session_features)[:, 1]
    label = clf.predict(session_features)
    session_features['Conversion_Prob'] = prob
    session_features['Predicted_Conversion'] = label
    return session_features

# Example usage with synthetic session
if __name__ == "__main__":
    # Example: single session
    example = pd.DataFrame({
        "session_depth": [3],
        "device_desktop": [1],
        "device_mobile": [0],
        "device_tablet": [0],
        "funnel_Landing": [1],
        "funnel_Product": [1],
        "funnel_Cart": [1],
        "funnel_Checkout": [0],
        "funnel_Purchase": [0],
        "tfidf_svd_0": [0.12],
        "tfidf_svd_1": [0.05],
        "tfidf_svd_2": [0.01],
        # Add more TF-IDF SVD components as per your data
    })
    pred_df = predict_conversion(example)
    print(pred_df)
