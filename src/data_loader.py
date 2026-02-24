import pandas as pd
from pathlib import Path

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "cleaned" / "ai4i2020_cleaned.csv"

def load_raw_data():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data not found at {RAW_DATA_PATH}")
    
    return pd.read_csv(RAW_DATA_PATH)

def load_cleaned_data():
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned data not found at {CLEANED_DATA_PATH}")
    
    return pd.read_csv(CLEANED_DATA_PATH)

def get_basic_info(df: pd.DataFrame):
    return {
        "shape": df.shape,
        "columns": df.columns.tolist()
    }