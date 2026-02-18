def add_engineered_features(df):
    df = df.copy()

    df["Temp_diff"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )

    df["Torque_RPM_ratio"] = (
        df["Torque [Nm]"] / (df["Rotational speed [rpm]"] + 1)
    )

    return df
