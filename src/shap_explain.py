# src/shap_explain.py
#
# Works with:
#   - Single tree estimators (RandomForest, GradientBoosting) → uses TreeExplainer (fast)
#   - VotingClassifier / CalibratedClassifier / any sklearn model → uses Explainer (model-agnostic)
#
# Both paths return the same output format so callers don't need to change.

from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap

from src.features import add_engineered_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.pkl"


def _load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )
    return joblib.load(MODEL_PATH)


def _is_tree_model(model) -> bool:
    """
    Returns True only for single estimators that TreeExplainer supports natively.
    VotingClassifier, CalibratedClassifierCV, stacking, etc. return False.
    """
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        ExtraTreesClassifier,
    )
    try:
        import xgboost
        xgb_types = (xgboost.XGBClassifier,)
    except ImportError:
        xgb_types = ()
    try:
        import lightgbm
        lgbm_types = (lightgbm.LGBMClassifier,)
    except ImportError:
        lgbm_types = ()

    supported = (
        RandomForestClassifier,
        GradientBoostingClassifier,
        ExtraTreesClassifier,
        *xgb_types,
        *lgbm_types,
    )
    return isinstance(model, supported)


def get_top_shap_drivers(sensor_row: dict, top_k: int = 8) -> list[dict]:
    """
    Compute SHAP values for a single sensor snapshot and return the top_k
    most influential features sorted by absolute SHAP magnitude.

    Returns:
        List of dicts: [{"feature": str, "shap_value": float}, ...]
    """
    pipeline = _load_pipeline()

    # Build input DataFrame and add engineered features
    df = pd.DataFrame([sensor_row])
    df = add_engineered_features(df)

    # Split pipeline into preprocessor and model
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]

    # Transform input
    X_trans = prep.transform(df)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    # Get feature names from preprocessor
    try:
        feature_names = list(prep.get_feature_names_out())
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_trans.shape[1])]

    # ----------------------------------------------------------------
    # Choose explainer based on model type
    # ----------------------------------------------------------------
    if _is_tree_model(model):
        # Fast path: native TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_trans)

        # Binary classification returns [class0_vals, class1_vals]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_vals = shap_values[0]

    else:
        # General path: model-agnostic KernelExplainer
        # We wrap predict_proba for class 1 (failure probability)
        def predict_fn(X):
            return model.predict_proba(X)[:, 1]

        # Use a zero background (fast single-row explanation)
        background = np.zeros((1, X_trans.shape[1]))
        explainer = shap.KernelExplainer(predict_fn, background)

        # nsamples=128 balances speed vs accuracy for a single row
        shap_values = explainer.shap_values(X_trans, nsamples=128, silent=True)
        shap_vals = shap_values[0]

    # ----------------------------------------------------------------
    # Pick top K by absolute magnitude
    # ----------------------------------------------------------------
    idx = np.argsort(np.abs(shap_vals))[::-1][:top_k]

    return [
        {
            "feature": str(feature_names[i]),
            "shap_value": float(shap_vals[i]),
        }
        for i in idx
    ]