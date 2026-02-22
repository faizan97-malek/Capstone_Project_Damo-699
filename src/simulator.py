# src/simulator.py

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

# Cache dataset in memory so we don't re-read it every refresh
_DATA_CACHE: pd.DataFrame | None = None


def _load_dataset() -> pd.DataFrame:
    """
    Load AI4I dataset for simulation (prefer cleaned, fallback to raw).
    Returns a DataFrame that includes Product ID + sensor columns.
    """
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE

    root = Path(__file__).resolve().parents[1]

    cleaned_path = root / "data" / "cleaned" / "ai4i2020_cleaned.csv"
    raw_path = root / "data" / "raw" / "ai4i2020.csv"

    # Try cleaned first
    if cleaned_path.exists():
        df = pd.read_csv(cleaned_path)
    elif raw_path.exists():
        df = pd.read_csv(raw_path)
    else:
        raise FileNotFoundError(
            "Could not find dataset. Expected one of:\n"
            f"- {cleaned_path}\n"
            f"- {raw_path}"
        )

    # If cleaned doesn't have Product ID, try raw and merge or just load raw
    if "Product ID" not in df.columns and raw_path.exists():
        raw_df = pd.read_csv(raw_path)

        # If cleaned has an obvious key (like 'UDI'), we can merge.
        # If not, we just switch to raw so Product ID is available.
        if "UDI" in df.columns and "UDI" in raw_df.columns:
            df = df.merge(raw_df[["UDI", "Product ID"]], on="UDI", how="left")
        else:
            df = raw_df

    required = [
        "Product ID",
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Simulator dataset is missing columns: {missing}")

    # Keep only what simulator needs (cleaner payload)
    df = df[required].dropna().reset_index(drop=True)

    _DATA_CACHE = df
    return _DATA_CACHE


def generate_sensor_state(seed: int | None = None) -> dict:
    """
    Sample ONE real row from the dataset and return it as a sensor snapshot.
    This includes Product ID and matches the model's expected input fields.
    """
    if seed is not None:
        random.seed(seed)

    df = _load_dataset()

    i = random.randrange(0, len(df))
    row = df.iloc[i].to_dict()

    # Make sure types are JSON-friendly
    row["Product ID"] = str(row["Product ID"])
    row["Type"] = str(row["Type"])
    for k in [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]:
        row[k] = float(row[k])

    return row


if __name__ == "__main__":
    print(generate_sensor_state())