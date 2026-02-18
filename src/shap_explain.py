# src/shap_explain.py

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap


from src.features import add_engineered_features


MODEL_PATH = Path("models") / "best_model.pkl"


def _load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run training first.")
    return joblib.load(MODEL_PATH)


def get_top_shap_drivers(sensor_row: dict, top_k: int = 8):
    """
    Returns the top SHAP drivers for ONE sensor reading.

    We must add engineered features first, because the saved pipeline was trained
    with engineered columns like Temp_diff and Torque_RPM_ratio.
    """

    pipeline = _load_pipeline()

    # Convert dict → DataFrame
    df = pd.DataFrame([sensor_row])

    # Add engineered features (IMPORTANT)
    df = add_engineered_features(df)

    # Split the pipeline
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]

    # Transform using the same preprocessor used in training
    X_trans = prep.transform(df)

    # If it returns sparse matrix, convert to numpy array
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    # SHAP explainer for tree models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)

    # For binary classification SHAP sometimes returns [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_vals = shap_values[0]

    # Get feature names from the preprocessor
    try:
        feature_names = prep.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(shap_vals))]

    # Pick top K by absolute SHAP magnitude
    idx = np.argsort(np.abs(shap_vals))[::-1][:top_k]

    drivers = []
    for i in idx:
        drivers.append({
            "feature": str(feature_names[i]),
            "shap_value": float(shap_vals[i])
        })

    return drivers