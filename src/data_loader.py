import pandas as pd
from pathlib import Path


# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "cleaned" / "ai4i2020_cleaned.csv"


def load_raw_data():
    """
    Load the raw AI4I 2020 dataset.

    Returns
    -------
    pd.DataFrame
        Raw dataset as a pandas DataFrame.
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data not found at {RAW_DATA_PATH}")
    
    return pd.read_csv(RAW_DATA_PATH)


def load_cleaned_data():
    """
    Load the cleaned AI4I 2020 dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset as a pandas DataFrame.
    """
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned data not found at {CLEANED_DATA_PATH}")
    
    return pd.read_csv(CLEANED_DATA_PATH)


def get_basic_info(df: pd.DataFrame):
    """
    Return basic dataset information for sanity checks.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset

    Returns
    -------
    dict
        Dictionary containing shape and column names
    """
    return {
        "shape": df.shape,
        "columns": df.columns.tolist()
    }
