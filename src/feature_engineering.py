import pandas as pd

def add_features(df):
    df["chol_per_age"] = df["chol"] / (df["age"] + 1)
    return df
