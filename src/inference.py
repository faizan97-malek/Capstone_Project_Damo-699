import joblib
import pandas as pd
from pathlib import Path

from src.features import add_engineered_features
from src.labeling import risk_tier

MODEL_PATH = Path("models") / "best_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError("Model not found. Run training first.")

model = joblib.load(MODEL_PATH)


def predict(sensor_row: dict, wear_penalty_strength: float = 0.0015) -> dict:
    """
    Accepts a single sensor snapshot (dict)
    Returns probability + risk label

    wear_penalty_strength:
        small additive penalty that increases with tool wear
        helps make the "risk over time" trend feel realistic
    """
    df = pd.DataFrame([sensor_row])
    df = add_engineered_features(df)

    # model probability
    prob = float(model.predict_proba(df)[0][1])

    # wear-based penalty (gentle)
    wear = float(sensor_row.get("Tool wear [min]", 0.0))
    penalty = wear * wear_penalty_strength / 200.0  # scaled down
    prob_adj = min(max(prob + penalty, 0.0), 1.0)

    return {
        "risk_probability": prob_adj,
        "risk_label": risk_tier(prob_adj, high_threshold=0.7),
    }


def compute_ttf_proxy(sensor_row: dict, wear_limit: float = 200.0) -> float:
    tool_wear = float(sensor_row.get("Tool wear [min]", 0))
    return float(max(wear_limit - tool_wear, 0))