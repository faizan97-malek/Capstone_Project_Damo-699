import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

from src.data_loader import load_raw_data
from src.features import add_engineered_features
from src.preprocessing import build_preprocessor

def train_model():

    print("Loading data...")
    df = load_raw_data()

    print("Adding engineered features...")
    df = add_engineered_features(df)

    # Target
    y = df["Machine failure"]
    X = df.drop(columns=["Machine failure"])

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Building preprocessing pipeline...")
    preprocessor = build_preprocessor(df)

    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        random_state=42
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    roc = roc_auc_score(y_test, y_proba)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc:.4f}")

    # Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "best_model.pkl"
    joblib.dump(pipeline, model_path)

    # Save evaluation metrics
    eval_path = models_dir / "eval.json"
    with open(eval_path, "w") as f:
        json.dump({
            "roc_auc": roc,
            "classification_report": report
        }, f, indent=4)

    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    train_model()
