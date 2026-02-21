from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(df):
    """
    Build preprocessing pipeline:
    - One-hot encode Type
    - Scale numeric sensor features

    IMPORTANT:
    We explicitly exclude target/label columns from features,
    including Machine failure and the individual failure type flags.
    """

    categorical_features = ["Type"]

    # Columns that should NEVER be used as features (targets / leakage)
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