def add_time_features(df):
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    return df

def add_time_features_teste_freqs(df, frequency="mon"):
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    if frequency == "day":
        df["day"] = df["time"].dt.day
        df["dayofyear"] = df["time"].dt.dayofyear
    return df