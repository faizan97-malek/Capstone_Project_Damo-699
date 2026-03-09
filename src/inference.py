from __future__ import annotations

import json as _json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from src.features import add_engineered_features
from src.labeling import risk_tier

ROOT = Path(__file__).resolve().parents[1]

# We keep all three model paths here so they are easy to find and update.
# best_model.pkl is the Soft Voting classifier, rul_regressor.joblib is
# the TTF regression model from notebook 09, and threshold.json stores
# the tuned decision threshold from train.py.
MODEL_PATH     = ROOT / "models" / "best_model.pkl"
RUL_MODEL_PATH = ROOT / "models" / "rul_regressor.joblib"
THRESHOLD_PATH = ROOT / "models" / "threshold.json"

# We set wear limits per machine type based on the failure distributions
# observed in notebook 09. Type H machines tend to fail around 240 min
# of tool wear, while M and L types last closer to 250 min.
_WEAR_LIMIT = {"H": 240, "M": 250, "L": 250}

# We load these once and cache them so we dont read the dataset on every call.
_rul_norm_constants: dict | None = None


def _load_rul_norm_constants() -> dict:
    # We need the max values of torque and RPM from the training data
    # because the RUL model was trained on normalized versions of these
    # features. If we use different max values here, the predictions
    # would be wrong.
    global _rul_norm_constants
    if _rul_norm_constants is not None:
        return _rul_norm_constants

    try:
        from src.data_loader import load_cleaned_data
        df = load_cleaned_data()
        torque_max = float(df["Torque [Nm]"].max())
        rpm_max    = float(df["Rotational speed [rpm]"].max())
    except Exception:
        # We hardcode these as a safety net in case the CSV is missing.
        # These values come from the AI4I 2020 dataset (10000 rows).
        torque_max = 76.6
        rpm_max    = 2886.0

    _rul_norm_constants = {
        "torque_max": torque_max,
        "rpm_max":    rpm_max,
    }
    return _rul_norm_constants


def prepare_rul_features(sensor_row: dict) -> pd.DataFrame:
    # We replicate the exact feature engineering from notebook 09 here
    # so that the RUL regression model receives the same columns it was
    # trained on: the 6 raw sensor features plus temp_diff, torque_norm,
    # wear_norm, and rpm_inv_norm.
    consts = _load_rul_norm_constants()

    mtype      = str(sensor_row.get("Type", "M")).upper()
    torque     = float(sensor_row.get("Torque [Nm]", 0.0))
    rpm        = float(sensor_row.get("Rotational speed [rpm]", 0.0))
    tool_wear  = float(sensor_row.get("Tool wear [min]", 0.0))
    air_temp   = float(sensor_row.get("Air temperature [K]", 0.0))
    proc_temp  = float(sensor_row.get("Process temperature [K]", 0.0))

    wear_limit = float(_WEAR_LIMIT.get(mtype, 250))

    row = {
        "Type":                      mtype,
        "Air temperature [K]":       air_temp,
        "Process temperature [K]":   proc_temp,
        "Rotational speed [rpm]":    rpm,
        "Torque [Nm]":               torque,
        "Tool wear [min]":           tool_wear,
        "temp_diff":                 proc_temp - air_temp,
        "torque_norm":               torque / consts["torque_max"] if consts["torque_max"] else 0.0,
        "wear_norm":                 tool_wear / wear_limit if wear_limit else 0.0,
        "rpm_inv_norm":              1.0 - (rpm / consts["rpm_max"]) if consts["rpm_max"] else 0.0,
    }

    return pd.DataFrame([row])


# We use lazy loading for all three models so the app only reads from
# disk the first time predict() or compute_ttf_proxy() is called.
# This avoids slowing down the import when other modules load inference.py.
_model = None
_rul_model = None
_threshold_loaded = False
FINAL_THRESHOLD = 0.5


def _load_models():
    global _model, _rul_model, FINAL_THRESHOLD, _threshold_loaded

    # We load the classification model (Soft Voting ensemble)
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}.\n"
                "Run:  python -m src.train"
            )
        _model = joblib.load(MODEL_PATH)

    # We load the tuned threshold from threshold.json which was saved
    # by train.py. This threshold was selected to achieve >= 95% recall.
    if not _threshold_loaded:
        if THRESHOLD_PATH.exists():
            with open(THRESHOLD_PATH) as f:
                FINAL_THRESHOLD = float(_json.load(f)["final_threshold"])
        _threshold_loaded = True

    # We load the RUL regression model if it exists. If the file is
    # missing, inference falls back to a simple wear rule instead.
    if _rul_model is None and RUL_MODEL_PATH.exists():
        _rul_model = joblib.load(RUL_MODEL_PATH)


def predict(sensor_row: dict) -> dict:
    # We predict failure risk for a single machine snapshot. The function
    # returns the probability, a human-readable risk label, and whether
    # the machine crosses the tuned decision threshold.
    _load_models()

    # We drop Product ID because it is not a model feature, but the
    # dashboard sometimes passes it in along with the sensor values.
    clean = {k: v for k, v in sensor_row.items() if k != "Product ID"}
    df = pd.DataFrame([clean])
    df = add_engineered_features(df)

    prob = float(_model.predict_proba(df)[0][1])

    return {
        "risk_probability": prob,
        "risk_label":       risk_tier(prob, high_threshold=0.70, medium_threshold=0.35),
        "is_high_risk":     prob >= FINAL_THRESHOLD,
        "threshold_used":   FINAL_THRESHOLD,
    }


def compute_ttf_proxy(sensor_row: dict, wear_limit: float = 250.0) -> dict:
    # We estimate time-to-failure in minutes using the trained regression
    # model from notebook 09. If that model is not available (file missing
    # or prediction fails), we fall back to a simple rule that subtracts
    # current tool wear from the wear limit.
    _load_models()

    # We try the regression model first because it accounts for all
    # sensor variables, not just tool wear.
    if _rul_model is not None:
        try:
            X_live = prepare_rul_features(sensor_row)
            pred   = float(_rul_model.predict(X_live)[0])
            pred   = max(pred, 0.0)
            return {
                "ttf_min": round(pred, 1),
                "method":  "regression_rul",
                "notes":   "TTF estimated using trained regression RUL model.",
            }
        except Exception as exc:
            import warnings
            warnings.warn(f"RUL regression failed, falling back to wear rule: {exc}")

    # We use this simple fallback when the regression model is missing
    # or throws an error. It only considers tool wear which is less
    # accurate but better than showing nothing.
    tool_wear = float(sensor_row.get("Tool wear [min]", 0.0))
    ttf = max(wear_limit - tool_wear, 0.0)
    return {
        "ttf_min": round(ttf, 1),
        "method":  "wear_rule_fallback",
        "notes":   "Fallback: wear_limit minus tool_wear (RUL model missing or failed).",
    }