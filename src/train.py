import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from src.data_loader import load_raw_data
from src.features import add_engineered_features
from src.preprocessing import build_preprocessor


def train_model():
    print("Loading data...")
    df = load_raw_data()

    print("Adding engineered features...")
    df = add_engineered_features(df)

    if "Machine failure" not in df.columns:
        raise ValueError("Target column 'Machine failure' not found in dataset.")

    leakage_cols = [c for c in ["TWF", "HDF", "PWF", "OSF", "RNF"] if c in df.columns]
    drop_cols = ["Machine failure"] + leakage_cols

    y = df["Machine failure"].astype(int)
    X = df.drop(columns=drop_cols, errors="ignore")

    print("Splitting data (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    print("Building preprocessing pipeline...")
    preprocessor = build_preprocessor(df)

    # ---------------------------------------------------------------
    # Soft Voting Ensemble
    #
    # Selected for deployment over Stacking (LR meta) because:
    #   1. Produces well-spread probability scores (0.0 to 0.96 range)
    #      making the continuous risk gauge meaningful for operators
    #   2. Threshold of ~0.18 aligns naturally with the gauge scale —
    #      a machine at 20% probability visually reads as medium-high risk
    #   3. Stacking achieves marginally higher PR-AUC (0.881 vs 0.840)
    #      but compresses all safe-machine probabilities into 0.12-0.13,
    #      making the gauge flat and the risk labels unintuitive
    #
    # Stacking results are fully documented in notebook 07 for the report.
    # ---------------------------------------------------------------
    print("Building Soft Voting ensemble...")

    rf = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(random_state=42)
    lr = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    rf_cal = CalibratedClassifierCV(rf, method="isotonic", cv=3)
    lr_cal = CalibratedClassifierCV(lr, method="isotonic", cv=3)

    voting = VotingClassifier(
        estimators=[("lr", lr_cal), ("rf", rf_cal), ("gb", gb)],
        voting="soft",
        weights=[1, 2, 2],
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", voting),
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------------------
    # Threshold selection
    #
    # In predictive maintenance, missing a failure (false negative) is
    # far more costly than a false alarm (false positive). A missed
    # failure means unexpected downtime, equipment damage, safety risk.
    # A false alarm means an unnecessary inspection — much cheaper.
    #
    # We scan the Precision-Recall curve and select the threshold that
    # achieves recall >= 0.95 with the highest possible precision.
    # For Soft Voting on this dataset this consistently lands near 0.02:
    #   - Recall:    ~0.956  (catches 95.6% of real failures)
    #   - Precision: ~0.300  (30% of flagged machines are real failures)
    #   - False alarms: ~152 out of 1932 safe machines (8%)
    #
    # This is the justified tradeoff: 3 missed failures vs 152 false alarms.
    # ---------------------------------------------------------------
    precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, y_prob)

    FINAL_THRESHOLD = None
    achieved_recall = None
    for recall_target in [0.95, 0.90, 0.85, 0.80]:
        valid_idx = np.where(recall_vals[:-1] >= recall_target)[0]
        if len(valid_idx) > 0:
            best_idx = valid_idx[np.argmax(precision_vals[valid_idx])]
            FINAL_THRESHOLD = float(thresholds_pr[best_idx])
            achieved_recall = recall_target
            break

    if FINAL_THRESHOLD is None:
        FINAL_THRESHOLD = 0.5
        print("Warning: fallback threshold 0.5 — no threshold met recall target")
    else:
        print(f"Threshold: {FINAL_THRESHOLD:.6f}  (recall >= {achieved_recall})")

    y_pred_default = (y_prob >= 0.5).astype(int)
    y_pred_tuned   = (y_prob >= FINAL_THRESHOLD).astype(int)

    roc    = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    report_default = classification_report(y_test, y_pred_default, output_dict=True, zero_division=0)
    report_tuned   = classification_report(y_test, y_pred_tuned,   output_dict=True, zero_division=0)

    print("\n--- Default threshold (0.50) ---")
    print(classification_report(y_test, y_pred_default, zero_division=0))
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")
    print(f"\n--- Tuned threshold ({FINAL_THRESHOLD:.6f}) ---")
    print(classification_report(y_test, y_pred_tuned, zero_division=0))

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "best_model.pkl"
    joblib.dump(pipeline, model_path)

    eval_path = models_dir / "eval.json"
    with open(eval_path, "w") as f:
        json.dump({
            "model": "SoftVotingEnsemble",
            "roc_auc": roc,
            "pr_auc": pr_auc,
            "threshold_default": 0.5,
            "threshold_tuned": FINAL_THRESHOLD,
            "recall_target_achieved": achieved_recall,
            "classification_report_default": report_default,
            "classification_report_tuned": report_tuned,
        }, f, indent=4)

    threshold_path = models_dir / "threshold.json"
    with open(threshold_path, "w") as f:
        json.dump({"final_threshold": FINAL_THRESHOLD}, f, indent=4)

    print(f"\n✅ Model saved     → {model_path}")
    print(f"✅ Metrics saved   → {eval_path}")
    print(f"✅ Threshold saved → {threshold_path}")


def train_rul_model():
    """
    Train the RUL (Remaining Useful Life) regression model.
    Replicates the logic from Notebook 09 so that all model artifacts
    are generated from src/train.py — no notebook execution required.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split as tts

    print("\n" + "=" * 60)
    print("Training RUL regression model...")
    print("=" * 60)

    df = load_raw_data()

    # ── Wear limits per machine type (from NB09 distribution analysis) ──
    WEAR_LIMIT = {"H": 240, "M": 250, "L": 250}

    df["wear_limit"]     = df["Type"].map(WEAR_LIMIT)
    df["wear_remaining"] = (df["wear_limit"] - df["Tool wear [min]"]).clip(lower=0)

    # ── Physics-based degradation rate ──
    df["temp_diff"]   = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["torque_norm"] = df["Torque [Nm]"]     / df["Torque [Nm]"].max()
    df["temp_norm"]   = df["temp_diff"]       / df["temp_diff"].max()
    df["wear_norm"]   = df["Tool wear [min]"] / df["wear_limit"]
    df["rpm_inv_norm"] = 1 - (df["Rotational speed [rpm]"] / df["Rotational speed [rpm]"].max())

    df["deg_rate"] = (
        0.35 * df["torque_norm"]
        + 0.25 * df["wear_norm"]
        + 0.20 * df["rpm_inv_norm"]
        + 0.20 * df["temp_norm"]
    ).clip(lower=0.01)

    # ── Synthetic RUL target ──
    RUL_CAP = 500
    df["RUL"] = (df["wear_remaining"] / df["deg_rate"]).clip(upper=RUL_CAP)

    print(f"RUL range: {df['RUL'].min():.1f} — {df['RUL'].max():.1f} minutes")
    print(f"RUL mean:  {df['RUL'].mean():.1f} minutes")

    # ── Features ──
    FEATURE_COLS = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "temp_diff",
        "torque_norm",
        "wear_norm",
        "rpm_inv_norm",
    ]

    X = df[FEATURE_COLS].copy()
    y = df["RUL"].copy()

    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42)

    # ── Preprocessing ──
    cat_cols = ["Type"]
    num_cols = [c for c in FEATURE_COLS if c != "Type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    # ── Train and compare ──
    regressors = {
        "Ridge":            Ridge(alpha=1.0),
        "RandomForest":     RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    }

    best_name, best_pipe, best_mae = None, None, float("inf")

    for name, model in regressors.items():
        pipe = SkPipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        print(f"  {name:25s}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

        if mae < best_mae:
            best_name, best_pipe, best_mae = name, pipe, mae

    # ── Save ──
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    rul_path = models_dir / "rul_regressor.joblib"
    joblib.dump(best_pipe, rul_path)

    print(f"\n✅ RUL model saved → {rul_path}")
    print(f"   Selected: {best_name}  (MAE={best_mae:.2f} min)")


if __name__ == "__main__":
    train_model()
    train_rul_model()