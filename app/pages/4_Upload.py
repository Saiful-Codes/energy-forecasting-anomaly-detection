import streamlit as st
import pandas as pd
import numpy as np

from utils import (
    load_model,
    load_threshold_config,
    get_threshold_value,
    backtest_forecast,
    compute_severity,
    validate_required_columns,
    prepare_uploaded_series,
)

st.title("4) Upload & Score")

st.write("Upload a CSV file and score it using the forecasting model + anomaly threshold.")

# ----------------------------
# Controls
# ----------------------------
horizon_label = st.selectbox("Scoring window", ["24 hours", "7 days"])
horizon = 24 if horizon_label == "24 hours" else 168

dt_col = st.text_input("Datetime column name", value="datetime")
y_col = st.text_input("Target column name", value="Global_active_power")

hourly_resample = st.checkbox("Resample to hourly (recommended)", value=True)

uploaded = st.file_uploader("Upload CSV", type=["csv"])

# ----------------------------
# Load assets (model + threshold)
# ----------------------------
model, model_err = load_model()
cfg, cfg_err = load_threshold_config()

if model_err:
    st.error(model_err)
    st.stop()

if cfg_err:
    st.error(cfg_err)
    st.stop()

threshold = get_threshold_value(cfg)
if threshold is None:
    st.error("Could not extract threshold value from threshold_v1.json.")
    st.json(cfg)
    st.stop()

st.caption(f"Using anomaly threshold: {threshold:.4f}")

# ----------------------------
# If no file uploaded, stop early
# ----------------------------
if uploaded is None:
    st.info("Upload a CSV to begin scoring.")
    st.stop()

# ----------------------------
# Read CSV
# ----------------------------
try:
    df = pd.read_csv(uploaded)
    if dt_col not in df.columns and y_col not in df.columns and df.shape[1] == 2:
        uploaded.seek(0)  # reset file pointer
        df = pd.read_csv(uploaded, header=None, names=[dt_col, y_col])
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.write("Preview:")
st.dataframe(df.head(20), use_container_width=True)

# ----------------------------
# Validate columns
# ----------------------------
missing = validate_required_columns(df, [dt_col, y_col])
if missing:
    st.error(f"Missing required columns: {missing}")
    st.write("Columns found:", list(df.columns))
    st.stop()

# ----------------------------
# Prepare series (datetime index, numeric, hourly resample)
# ----------------------------
try:
    series = prepare_uploaded_series(df, dt_col=dt_col, y_col=y_col, hourly=hourly_resample)
except Exception as e:
    st.error(f"Could not prepare time series: {e}")
    st.stop()

if series.empty:
    st.error("After cleaning, no valid rows remain. Check datetime parsing and numeric values.")
    st.stop()

st.success(f"Prepared time series: {series.shape[0]:,} rows")
st.caption(f"Date range: {series.index.min()} to {series.index.max()}")

# ----------------------------
# Run scoring (backtest forecast)
# ----------------------------
result, err = backtest_forecast(model, series, horizon)
if err:
    st.error(err)
    st.info("Tip: Upload more history (especially for 7 days). The model needs lag history to predict.")
    st.stop()

pred_index, y_true, y_pred = result

residuals = (y_true - y_pred)
abs_resid = np.abs(residuals)
is_anomaly = abs_resid > threshold
severity = compute_severity(residuals, threshold)

# ----------------------------
# Report table
# ----------------------------
report = pd.DataFrame({
    "datetime": pred_index,
    "actual": y_true,
    "forecast": y_pred,
    "residual": residuals,
    "abs_residual": abs_resid,
    "threshold": threshold,
    "is_anomaly": is_anomaly,
    "severity": severity,
})

# Summary
count = int(report["is_anomaly"].sum())
total = int(report.shape[0])
rate = (count / total) * 100.0 if total else 0.0

st.subheader("Scoring Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Rows scored", f"{total:,}")
c2.metric("Anomalies flagged", f"{count:,}")
c3.metric("Anomaly rate", f"{rate:.1f}%")

# Show only anomalies table first (professional)
st.subheader("Flagged anomalies")
anoms = report[report["is_anomaly"]].copy().sort_values("severity", ascending=False)

if anoms.empty:
    st.info("No anomalies detected in the scoring window.")
else:
    st.dataframe(
        anoms[["datetime", "actual", "forecast", "residual", "abs_residual", "severity"]],
        use_container_width=True
    )

# Full report
with st.expander("See full scored report"):
    st.dataframe(report, use_container_width=True)

# Download report
csv = report.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download scored report (CSV)",
    data=csv,
    file_name=f"scored_report_{horizon}h.csv",
    mime="text/csv"
)