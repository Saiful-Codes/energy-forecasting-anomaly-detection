import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils import (
    load_model,
    load_threshold_config,
    load_default_data,
    infer_datetime_col,
    infer_target_col,
    ensure_datetime_index,
    backtest_forecast,
    get_threshold_value,
    compute_severity,
)

st.title("3) Anomalies")


# Controls
window_label = st.selectbox("Analysis window", ["24 hours", "7 days"])
horizon = 24 if window_label == "24 hours" else 168

show_band = st.checkbox("Show uncertainty band (±1.96 * residual std)", value=False)
min_severity = st.slider("Minimum severity to display", min_value=1.0, max_value=5.0, value=1.0, step=0.1)


# Load assets
model, model_err = load_model()
cfg, cfg_err = load_threshold_config()
df, df_err = load_default_data()

if model_err:
    st.error(model_err)
    st.stop()

if cfg_err:
    st.error(cfg_err)
    st.stop()

if df_err:
    st.error(df_err)
    st.stop()

threshold = get_threshold_value(cfg)
if threshold is None:
    st.error("Could not extract threshold value from threshold_v1.json. Show me the JSON keys and I’ll map it.")
    st.json(cfg)
    st.stop()


# Infer columns
dt_col = infer_datetime_col(df)
y_col = infer_target_col(df)

if dt_col is None:
    st.error("Could not find datetime column. Rename your datetime column to 'datetime' (recommended).")
    st.write("Columns found:", list(df.columns))
    st.stop()

if y_col is None:
    st.error("Could not infer target column. Rename your target column to 'Global_active_power' (recommended).")
    st.write("Columns found:", list(df.columns))
    st.stop()


# Prepare time series
df_ts = ensure_datetime_index(df, dt_col)
y = df_ts[y_col].astype(float).dropna()


# Backtest to get residuals
result, err = backtest_forecast(model, y, horizon)
if err:
    st.error(err)
    st.stop()

pred_index, y_true, y_pred = result

residuals = (y_true - y_pred)
abs_resid = np.abs(residuals)

# Anomaly rule: |residual| > threshold
is_anomaly = abs_resid > threshold
severity = compute_severity(residuals, threshold)

# Apply severity filter
is_anomaly = is_anomaly & (severity >= min_severity)


# Summary
total = len(pred_index)
count = int(is_anomaly.sum())
rate = (count / total) * 100.0 if total else 0.0

st.caption(
    f"Window: {pred_index[0]} to {pred_index[-1]} | "
    f"Threshold: {threshold:.4f} | "
    f"Anomalies: {count}/{total} ({rate:.1f}%)"
)


# Optional uncertainty band (same logic as forecast page)
res_std = float(pd.Series(residuals).dropna().std(ddof=0))
upper = y_pred + 1.96 * res_std
lower = y_pred - 1.96 * res_std


# Plot: actual vs forecast + anomaly points highlighted
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(pred_index, y_true, label="Actual", linewidth=2)
ax.plot(pred_index, y_pred, label="Forecast", linewidth=2)

if show_band:
    ax.fill_between(pred_index, lower, upper, alpha=0.12, label="Uncertainty band")

# Highlight anomaly points on the actual series
anomaly_times = pred_index[is_anomaly]
anomaly_vals = np.array(y_true)[is_anomaly]

ax.scatter(anomaly_times, anomaly_vals, s=40, label="Anomaly", zorder=5)

ax.set_title(f"Anomaly Detection ({window_label})")
ax.set_ylabel(y_col)
ax.set_xlabel("Time")

fig.autofmt_xdate(rotation=30)
ax.margins(x=0)
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
st.pyplot(fig, clear_figure=True)


# Severity table (professional)
out = pd.DataFrame({
    "datetime": pred_index,
    "actual": y_true,
    "forecast": y_pred,
    "residual": residuals,
    "abs_residual": abs_resid,
    "is_anomaly": is_anomaly,
    "severity": severity,
})

anoms = out[out["is_anomaly"]].copy()
anoms = anoms.sort_values("severity", ascending=False)

st.subheader("Severity table")

if len(anoms) == 0:
    st.info("No anomalies detected in this window with the current settings.")
else:
    st.dataframe(
        anoms[["datetime", "actual", "forecast", "residual", "abs_residual", "severity"]],
        use_container_width=True
    )

    # Download anomalies only
    csv = anoms.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download anomalies (CSV)",
        data=csv,
        file_name=f"anomalies_{horizon}h.csv",
        mime="text/csv"
    )

with st.expander("See full window table"):
    st.dataframe(out, use_container_width=True)