from __future__ import annotations

import json as _json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from src.features import add_engineered_features
from src.labeling import risk_tier

# ---------------------------------------------------------------------------
# Paths (anchored to project root)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH     = ROOT / "models" / "best_model.pkl"          # Soft Voting classification
RUL_MODEL_PATH = ROOT / "models" / "rul_regressor.joblib"    # Regression RUL (NB09)
THRESHOLD_PATH = ROOT / "models" / "threshold.json"

# ---------------------------------------------------------------------------
# RUL feature-engineering constants
#
# These replicate the exact logic from NB09 cell 11.
# The normalization denominators (TORQUE_MAX, RPM_MAX) are the .max() of the
# full AI4I 2020 cleaned dataset — the same values NB09 used to build the
# features before training the pipeline.
#
# If you retrain on a different dataset, update these or (better) save them
# as a JSON artifact from the notebook and load them here.
# ---------------------------------------------------------------------------
_WEAR_LIMIT = {"H": 240, "M": 250, "L": 250}

# Lazy-loaded from the dataset once, then cached
_rul_norm_constants: dict | None = None


def _load_rul_norm_constants() -> dict:
    """
    Compute the normalization max values from the cleaned dataset.
    These MUST match what NB09 used when it created torque_norm, rpm_inv_norm, etc.
    Loaded once and cached for the process lifetime.
    """
    global _rul_norm_constants
    if _rul_norm_constants is not None:
        return _rul_norm_constants

    try:
        from src.data_loader import load_cleaned_data
        df = load_cleaned_data()
        torque_max = float(df["Torque [Nm]"].max())
        rpm_max    = float(df["Rotational speed [rpm]"].max())
    except Exception:
        # Hardcoded fallbacks from the AI4I 2020 dataset (10 000 rows)
        torque_max = 76.6
        rpm_max    = 2886.0

    _rul_norm_constants = {
        "torque_max": torque_max,
        "rpm_max":    rpm_max,
    }
    return _rul_norm_constants


def prepare_rul_features(sensor_row: dict) -> pd.DataFrame:
    """
    Replicate NB09's feature engineering for a single sensor snapshot
    so the RUL regression pipeline receives the columns it was trained on:

        Type, Air temperature [K], Process temperature [K],
        Rotational speed [rpm], Torque [Nm], Tool wear [min],
        temp_diff, torque_norm, wear_norm, rpm_inv_norm
    """
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


# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------
_model = None
_rul_model = None
_threshold_loaded = False
FINAL_THRESHOLD = 0.5  # default until threshold.json is read


def _load_models():
    global _model, _rul_model, FINAL_THRESHOLD, _threshold_loaded

    # --- Classification model (Soft Voting) ---
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}.\n"
                "Run:  python -m src.train"
            )
        _model = joblib.load(MODEL_PATH)

    # --- Tuned threshold ---
    if not _threshold_loaded:
        if THRESHOLD_PATH.exists():
            with open(THRESHOLD_PATH) as f:
                FINAL_THRESHOLD = float(_json.load(f)["final_threshold"])
        _threshold_loaded = True

    # --- RUL regression model (optional) ---
    if _rul_model is None and RUL_MODEL_PATH.exists():
        _rul_model = joblib.load(RUL_MODEL_PATH)


# ---------------------------------------------------------------------------
# Classification — failure risk
# ---------------------------------------------------------------------------
def predict(sensor_row: dict) -> dict:
    """
    Predict failure risk for a single sensor snapshot.

    Parameters
    ----------
    sensor_row : dict
        Keys must include Type, Air temperature [K], Process temperature [K],
        Rotational speed [rpm], Torque [Nm], Tool wear [min].
        Product ID (if present) is ignored automatically.

    Returns
    -------
    dict with:
        risk_probability  – float, model P(failure)
        risk_label        – str,   "Low" / "Medium" / "High"
        is_high_risk      – bool,  probability >= tuned threshold
        threshold_used    – float, the tuned decision threshold
    """
    _load_models()

    # Drop non-feature keys the dashboard may pass in
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


# ---------------------------------------------------------------------------
# RUL / TTF estimation
# ---------------------------------------------------------------------------
def compute_ttf_proxy(sensor_row: dict, wear_limit: float = 250.0) -> dict:
    """
    Return a Time-to-Failure (TTF) / Remaining Useful Life estimate in minutes.

    Primary path  : RUL regression model (rul_regressor.joblib) with proper
                    feature engineering matching NB09.
    Fallback path : simple  wear_limit − tool_wear  rule.

    Parameters
    ----------
    sensor_row : dict
        Same format as predict().
    wear_limit : float
        Used only by the fallback rule.

    Returns
    -------
    dict with:
        ttf_min  – float, estimated minutes to failure
        method   – str,   "regression_rul" or "wear_rule_fallback"
        notes    – str,   human-readable explanation
    """
    _load_models()

    # ---- 1) Regression-based RUL (preferred) ----
    if _rul_model is not None:
        try:
            X_live = prepare_rul_features(sensor_row)
            pred   = float(_rul_model.predict(X_live)[0])
            pred   = max(pred, 0.0)
            return {
                "ttf_min": round(pred, 1),
                "method":  "regression_rul",
                "notes":   "TTF estimated using trained regression RUL model (NB09).",
            }
        except Exception as exc:
            # Log the real error so it's debuggable, then fall through
            import warnings
            warnings.warn(f"RUL regression failed, falling back to wear rule: {exc}")

    # ---- 2) Fallback: simple wear rule ----
    tool_wear = float(sensor_row.get("Tool wear [min]", 0.0))
    ttf = max(wear_limit - tool_wear, 0.0)
    return {
        "ttf_min": round(ttf, 1),
        "method":  "wear_rule_fallback",
        "notes":   "Fallback: wear_limit − tool_wear (RUL model missing or failed).",
    }