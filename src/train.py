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


# Tuned threshold from your notebook (recall >= 0.80)
FINAL_THRESHOLD = 0.184518


def train_model():
    print("Loading data...")
    df = load_raw_data()

    print("Adding engineered features...")
    df = add_engineered_features(df)

    # Target / Features
    if "Machine failure" not in df.columns:
        raise ValueError("Target column 'Machine failure' not found in dataset.")

    # Drop leakage columns if present
    leakage_cols = [c for c in ["TWF", "HDF", "PWF", "OSF", "RNF"] if c in df.columns]
    drop_cols = ["Machine failure"] + leakage_cols

    y = df["Machine failure"].astype(int)
    X = df.drop(columns=drop_cols, errors="ignore")

    print("Splitting data (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Building preprocessing pipeline...")
    preprocessor = build_preprocessor(df)  # if your function expects df, keep as-is

    # -----------------------------
    # Soft Voting Ensemble
    # -----------------------------
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

    # Calibrate models that benefit from better probabilities (LR, RF)
    rf_cal = CalibratedClassifierCV(rf, method="isotonic", cv=3)
    lr_cal = CalibratedClassifierCV(lr, method="isotonic", cv=3)

    voting = VotingClassifier(
        estimators=[("lr", lr_cal), ("rf", rf_cal), ("gb", gb)],
        voting="soft",
        weights=[1, 2, 2],
    )

    pipeline = Pipeline(
        [
            ("prep", preprocessor),
            ("model", voting),
        ]
    )

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred_default = (y_prob >= 0.5).astype(int)
    y_pred_tuned = (y_prob >= FINAL_THRESHOLD).astype(int)

    # Metrics
    roc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    report_default = classification_report(y_test, y_pred_default, output_dict=True, zero_division=0)
    report_tuned = classification_report(y_test, y_pred_tuned, output_dict=True, zero_division=0)

    print("\n--- Default threshold (0.50) ---")
    print(classification_report(y_test, y_pred_default, zero_division=0))

    print(f"\nROC-AUC: {roc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")

    print(f"\n--- Tuned threshold ({FINAL_THRESHOLD}) ---")
    print(classification_report(y_test, y_pred_tuned, zero_division=0))

    # Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "best_model_soft_voting.joblib"
    joblib.dump(pipeline, model_path)

    # Save evaluation metrics
    eval_path = models_dir / "eval_soft_voting.json"
    with open(eval_path, "w") as f:
        json.dump(
            {
                "roc_auc": roc,
                "pr_auc": pr_auc,
                "threshold_default": 0.5,
                "threshold_tuned": FINAL_THRESHOLD,
                "classification_report_default": report_default,
                "classification_report_tuned": report_tuned,
            },
            f,
            indent=4,
        )

    # Save threshold separately (optional, convenient for inference)
    threshold_path = models_dir / "threshold.json"
    with open(threshold_path, "w") as f:
        json.dump({"final_threshold": FINAL_THRESHOLD}, f, indent=4)

    print(f"\nModel saved to {model_path}")
    print(f"Metrics saved to {eval_path}")
    print(f"Threshold saved to {threshold_path}")


if __name__ == "__main__":
    train_model()