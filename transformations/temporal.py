def add_time_features(df):
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    return df
