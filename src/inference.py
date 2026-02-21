# src/inference.py

import joblib
import pandas as pd
from pathlib import Path

from src.features import add_engineered_features
from src.labeling import risk_tier


# Load trained model once
MODEL_PATH = Path("models") / "best_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError("Model not found. Run training first.")

model = joblib.load(MODEL_PATH)


def predict(sensor_row: dict) -> dict:
    """
    Accepts a single machine sensor reading (dictionary)
    Returns probability and risk label
    """

    # Convert dictionary → DataFrame
    df = pd.DataFrame([sensor_row])

    # Add engineered features
    df = add_engineered_features(df)

    # Predict probability of failure (class 1)
    probability = model.predict_proba(df)[0][1]

    result = {
        "risk_probability": float(probability),
        "risk_label": risk_tier(probability, high_threshold=0.7)
    }

    return result


def compute_ttf_proxy(sensor_row: dict, wear_limit: float = 200.0) -> float:
    """
    Simple Remaining Useful Life proxy based on tool wear.
    This is NOT survival analysis.
    """

    tool_wear = sensor_row.get("Tool wear [min]", 0)

    remaining_time = max(wear_limit - tool_wear, 0)

    return float(remaining_time)
