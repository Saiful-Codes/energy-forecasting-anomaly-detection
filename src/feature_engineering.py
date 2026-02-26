import numpy as np
import pandas as pd

TARGET_COL = "Global_active_power"

FEATURE_COLS = [
    "dayofweek",
    "month",
    "hour_sin",
    "hour_cos",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_std_24",
    "rolling_mean_168",
]

def make_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """
    Build Feature Engineering v2 features for hourly energy forecasting.
    input:
      df indexed by datetime (DatetimeIndex) and df contains the target column (default: Global_active_power)
    Returns:
      X features DataFrame with FEATURE_COLS and Rows containing NaNs caused by lags/rolling windows are dropped
    """
    df_feat = df.copy()

    # base time features
    df_feat["hour"] = df_feat.index.hour
    df_feat["dayofweek"] = df_feat.index.dayofweek
    df_feat["month"] = df_feat.index.month

    # lag features
    df_feat["lag_1"] = df_feat[target_col].shift(1)
    df_feat["lag_24"] = df_feat[target_col].shift(24)
    df_feat["lag_168"] = df_feat[target_col].shift(168)

    # rolling features (shift then roll to avoid leakage)
    df_feat["rolling_mean_24"] = df_feat[target_col].shift(1).rolling(24).mean()
    df_feat["rolling_std_24"] = df_feat[target_col].shift(1).rolling(24).std()
    df_feat["rolling_mean_168"] = df_feat[target_col].shift(1).rolling(168).mean()

    # cyclical hour encoding
    df_feat["hour_sin"] = np.sin(2 * np.pi * df_feat["hour"] / 24)
    df_feat["hour_cos"] = np.cos(2 * np.pi * df_feat["hour"] / 24)

    # drop NaNs created by lags/rolling
    df_feat = df_feat.dropna()

    X = df_feat[FEATURE_COLS].copy()
    return X
