import joblib
import pandas as pd
from pathlib import Path

from src.features import add_engineered_features
from src.labeling import risk_tier

# Use the exported ensemble model
MODEL_PATH = Path("models") / "best_model_soft_voting.joblib"

# Tuned threshold to meet recall >= 0.80 (from your notebook)
FINAL_THRESHOLD = 0.184518

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Run the ensemble notebook and export the model first."
    )

model = joblib.load(MODEL_PATH)


def predict(sensor_row: dict) -> dict:
    """
    Predict failure risk for a single sensor snapshot.

    Returns:
      - risk_probability: model probability of failure
      - risk_label: tier label (based on your risk_tier function)
      - is_high_risk: boolean using tuned threshold (recall-focused)
    """
    df = pd.DataFrame([sensor_row])
    df = add_engineered_features(df)

    prob = float(model.predict_proba(df)[0][1])

    return {
        "risk_probability": prob,
        "risk_label": risk_tier(prob, high_threshold=0.7),   # keep your tiers if you like
        "is_high_risk": prob >= FINAL_THRESHOLD,            # this is your recall-optimized decision
        "threshold_used": FINAL_THRESHOLD,
    }


def compute_ttf_proxy(sensor_row: dict, wear_limit: float = 200.0) -> float:
    tool_wear = float(sensor_row.get("Tool wear [min]", 0))
    return float(max(wear_limit - tool_wear, 0))