import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Builds the preprocessing block for our ML pipeline.

    We do two things:
    1) One-hot encode the machine 'Type' column
    2) Standard-scale numeric sensor + engineered features

    IMPORTANT:
    The dataset also contains ID-like columns (e.g., 'Product ID', 'UDI').
    Those should NOT be used for modeling, so we explicitly drop them.
    """

    # 1) Categorical column
    categorical_features = ["Type"]

    # 2) Columns we never want in features
    #    - target column
    #    - ID columns (pure identifiers → no predictive meaning)
    drop_cols = ["Machine failure", "Product ID", "UDI"]

    # 3) Detect numeric features safely (avoid strings like 'M18918')
    numeric_features = []
    for col in df.columns:
        if col in drop_cols or col in categorical_features:
            continue

        # Only keep real numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)

    # Build ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    return preprocessor