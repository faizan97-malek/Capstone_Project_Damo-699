from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(df):
    categorical_features = ["Type"]

    exclude_cols = {
        "UDI",
        "Product ID",
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    }

    numeric_features = [
        col
        for col in df.columns
        if col not in categorical_features and col not in exclude_cols
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    return preprocessor


def risk_tier(
    probability: float,
    high_threshold: float = 0.70,
    medium_threshold: float = 0.35,
) -> str:
    """
    Convert failure probability into Low / Medium / High risk label.
    """

    if probability >= high_threshold:
        return "High Risk"
    elif probability >= medium_threshold:
        return "Medium Risk"
    else:
        return "Low Risk"