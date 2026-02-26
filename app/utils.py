import os
import json
import joblib
import pandas as pd
import streamlit as st
import numpy as np

# --- Paths (relative to project root) ---
DEFAULT_DATA_PATH = "data/processed/hourly_energy.csv"

# Change these names if your saved model file is named differently
MODEL_PATH = "models/forecast_pipeline_v1.joblib"
THRESHOLD_CONFIG_PATH = "models/threshold_v1.json" # from Week 2 (committed)

@st.cache_resource
def load_model():
    """
    Loads the trained forecasting model pipeline.
    Note: model artifact might be gitignored but should exist locally.
    """
    if not os.path.exists(MODEL_PATH):
        return None, f"Missing model file at: {MODEL_PATH}"
    model = joblib.load(MODEL_PATH)
    return model, None

@st.cache_data
def load_threshold_config():
    """
    Loads threshold config (P95 etc.) from JSON committed in Week 2.
    """
    if not os.path.exists(THRESHOLD_CONFIG_PATH):
        return None, f"Missing threshold config at: {THRESHOLD_CONFIG_PATH}"
    with open(THRESHOLD_CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    return cfg, None

@st.cache_data
def load_default_data():
    """
    Loads the processed hourly dataset you created in Week 1.
    """
    if not os.path.exists(DEFAULT_DATA_PATH):
        return None, f"Missing data file at: {DEFAULT_DATA_PATH}"
    df = pd.read_csv(DEFAULT_DATA_PATH)

    # If you saved datetime column under a name like 'datetime' or 'DateTime',
    # you can parse it here later. For skeleton, keep it simple.
    return df, None


def infer_datetime_col(df):
    candidates = ["datetime", "Datetime", "DateTime", "timestamp", "ts", "date"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def infer_target_col(df):
    candidates = ["Global_active_power", "target", "y", "value"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: if only 1 numeric column exists, assume it's the target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    return None

def ensure_datetime_index(df, dt_col):
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col)
    df = df.set_index(dt_col)
    return df

def mae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def make_features_from_series(series, t, include_cyclical=True):
    """
    Build one-row feature vector for timestamp t using the history in `series`.

    Assumptions (based on Week 2):
    - hourly data
    - features include: hour, dayofweek, month
    - lags: 1, 24, 168
    - rolling mean/std: 24, 168

    series: pandas Series with DateTimeIndex, values are target (y)
    t: timestamp to predict (must be next hour after last known)
    """
    # Helper to safely get historical value
    def val_at(ts):
        return float(series.loc[ts]) if ts in series.index else np.nan

    # Required history points
    lag_1 = val_at(t - pd.Timedelta(hours=1))
    lag_24 = val_at(t - pd.Timedelta(hours=24))
    lag_168 = val_at(t - pd.Timedelta(hours=168))

    # Rolling windows end at t-1 (past only)
    hist = series.loc[: t - pd.Timedelta(hours=1)]
    roll_mean_24 = float(hist.tail(24).mean()) if len(hist) >= 24 else np.nan
    roll_std_24 = float(hist.tail(24).std(ddof=0)) if len(hist) >= 24 else np.nan
    roll_mean_168 = float(hist.tail(168).mean()) if len(hist) >= 168 else np.nan

    hour = t.hour
    dayofweek = t.dayofweek
    month = t.month

    row = {
        "hour": hour,
        "dayofweek": dayofweek,
        "month": month,
        "lag_1": lag_1,
        "lag_24": lag_24,
        "lag_168": lag_168,
        "rolling_mean_24": roll_mean_24,
        "rolling_std_24": roll_std_24,
        "rolling_mean_168": roll_mean_168,
    }

    if include_cyclical:
        row["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        row["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    return pd.DataFrame([row])

def recursive_forecast(model, y_hist, start_t, horizon):
    series = y_hist.copy()
    preds = []
    idx = []

    expected = get_expected_features(model)

    t = start_t
    for _ in range(horizon):
        X = make_features_from_series(series, t)

        # Align columns to what model expects
        if expected is not None:
            X = X.reindex(columns=expected)

        yhat = float(model.predict(X)[0])
        preds.append(yhat)
        idx.append(t)

        series.loc[t] = yhat
        t = t + pd.Timedelta(hours=1)

    return pd.DatetimeIndex(idx), np.array(preds)


def backtest_forecast(model, y_series, horizon):
    """
    Backtest: take a window at the end of the series of length `horizon`.
    Use data BEFORE that window as history, and predict that window.
    """
    if len(y_series) < (horizon + 168 + 24 + 2):
        return None, "Not enough history for 7d backtest. Need at least ~10+ days."

    y_series = y_series.dropna().copy()
    test_start = y_series.index[-horizon]
    hist = y_series.loc[: test_start - pd.Timedelta(hours=1)]
    actual = y_series.loc[test_start : y_series.index[-1]]

    pred_index, preds = recursive_forecast(model, hist, test_start, horizon)
    actual = actual.reindex(pred_index)

    return (pred_index, actual.values, preds), None

def get_expected_features(model):
    """
    Returns the feature names the model was trained with.
    Works for pipelines or plain estimators.
    """
    # If it's a Pipeline, the final estimator is usually the last step
    if hasattr(model, "named_steps"):
        last_step = list(model.named_steps.values())[-1]
        if hasattr(last_step, "feature_names_in_"):
            return list(last_step.feature_names_in_)
    # If it's an estimator
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None

def get_threshold_value(cfg: dict) -> float:
    if cfg is None or not isinstance(cfg, dict):
        return None

    # 1) Direct common keys
    for key in ["threshold", "value", "residual_threshold", "thr", "thr_p95", "thr_p99", "p95", "p99"]:
        if key in cfg and isinstance(cfg[key], (int, float)):
            return float(cfg[key])

    # 2) Your format: params -> thr_p95 / thr_p99 / threshold
    params = cfg.get("params")
    if isinstance(params, dict):
        for key in ["thr_p95", "thr_p99", "threshold", "value", "thr"]:
            if key in params and isinstance(params[key], (int, float)):
                return float(params[key])

        # If strategy says p95/p99, map to the correct param key
        strategy = cfg.get("strategy")
        if isinstance(strategy, str):
            strategy = strategy.lower().strip()
            if strategy == "p95" and "thr_p95" in params:
                return float(params["thr_p95"])
            if strategy == "p99" and "thr_p99" in params:
                return float(params["thr_p99"])

    # 3) Percentiles nested
    percentiles = cfg.get("percentiles")
    if isinstance(percentiles, dict):
        for key in ["p95", "p99"]:
            if key in percentiles and isinstance(percentiles[key], (int, float)):
                return float(percentiles[key])

    return None

def compute_severity(residuals: np.ndarray, threshold: float) -> np.ndarray:
    """
    Severity score = how many times bigger the |residual| is than threshold.
    severity = |residual| / threshold
    """
    abs_r = np.abs(residuals)
    return abs_r / max(threshold, 1e-8)

def top_percent_threshold(series: pd.Series, pct: float) -> float:
    """
    Returns threshold value for the top pct% highest values.
    Example: pct=5 -> returns 95th percentile.
    """
    q = 1.0 - (pct / 100.0)
    return float(series.quantile(q))

def validate_required_columns(df: pd.DataFrame, required_cols: list) -> list:
    missing = [c for c in required_cols if c not in df.columns]
    return missing

def parse_datetime_column(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    out = df.copy()
    out[dt_col] = pd.to_datetime(out[dt_col], errors="coerce")
    out = out.dropna(subset=[dt_col])
    return out

def prepare_uploaded_series(df: pd.DataFrame, dt_col: str, y_col: str, hourly=True) -> pd.Series:
    """
    Returns a clean time series (pd.Series) with DateTimeIndex.
    If hourly=True, resamples to hourly mean.
    """
    out = df[[dt_col, y_col]].copy()
    out = parse_datetime_column(out, dt_col)
    out = out.sort_values(dt_col).set_index(dt_col)

    # Convert numeric safely
    out[y_col] = pd.to_numeric(out[y_col], errors="coerce")
    out = out.dropna(subset=[y_col])

    series = out[y_col]

    if hourly:
        series = series.resample("H").mean()

    return series