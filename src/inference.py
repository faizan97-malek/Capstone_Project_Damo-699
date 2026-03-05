from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path

from src.features import add_engineered_features
from src.labeling import risk_tier

# ---------------------------
# Paths (anchor to project root)
# ---------------------------
ROOT = Path(__file__).resolve().parents[1]

# Classification model (Soft Voting Ensemble)
MODEL_PATH = ROOT / "models" / "best_model.pkl"

# Regression RUL model (created in your notebook)
RUL_MODEL_PATH = ROOT / "models" / "rul_regressor.joblib"

# Tuned threshold — loaded dynamically from models/threshold.json (written by train.py)

import json as _json

# ---------------------------
# Lazy model loading
# ---------------------------
_model = None
_rul_model = None
_threshold_loaded = False
FINAL_THRESHOLD = 0.5  # default until loaded


def _load_models():
    global _model, _rul_model, FINAL_THRESHOLD, _threshold_loaded

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train.py first."
            )
        _model = joblib.load(MODEL_PATH)

    if not _threshold_loaded:
        _threshold_path = ROOT / "models" / "threshold.json"
        if _threshold_path.exists():
            FINAL_THRESHOLD = float(_json.load(open(_threshold_path))["final_threshold"])
        _threshold_loaded = True

    if _rul_model is None and RUL_MODEL_PATH.exists():
        _rul_model = joblib.load(RUL_MODEL_PATH)


def predict(sensor_row: dict) -> dict:
    """
    Predict failure risk for a single sensor snapshot.

    Returns:
      - risk_probability: model probability of failure
      - risk_label: tier label (based on your risk_tier function)
      - is_high_risk: boolean using tuned threshold (recall-focused)
    """
    _load_models()

    df = pd.DataFrame([sensor_row])
    df = add_engineered_features(df)

    prob = float(_model.predict_proba(df)[0][1])

    return {
        "risk_probability": prob,
        "risk_label": risk_tier(prob, high_threshold=0.70, medium_threshold=0.35),
        "is_high_risk": prob >= FINAL_THRESHOLD,
        "threshold_used": FINAL_THRESHOLD,
    }


def compute_ttf_proxy(sensor_row: dict, wear_limit: float = 250.0) -> dict:
    """
    Return TTF / RUL estimate in minutes.

    Primary: regression model (rul_regressor.joblib) if available
    Fallback: wear_limit - tool_wear
    """
    _load_models()

    # 1) Regression-based RUL (preferred)
    if _rul_model is not None:
        try:
            X_live = pd.DataFrame([sensor_row])
            pred = float(_rul_model.predict(X_live)[0])
            pred = float(max(pred, 0.0))
            return {
                "ttf_min": round(pred, 1),
                "method": "regression_rul",
                "notes": "TTF estimated using trained regression RUL model."
            }
        except Exception:
            pass

    # 2) Fallback: simple wear rule
    tool_wear = float(sensor_row.get("Tool wear [min]", 0.0))
    ttf = float(max(wear_limit - tool_wear, 0.0))
    return {
        "ttf_min": round(ttf, 1),
        "method": "wear_rule_fallback",
        "notes": "Fallback: wear_limit - tool_wear (RUL model missing or failed)."
    }