import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from utils import (
    load_model,
    load_default_data,
    infer_datetime_col,
    infer_target_col,
    ensure_datetime_index,
    backtest_forecast,
    mae, rmse, mape,
    get_expected_features
)

st.title("2) Forecast")

# ----------------------------
# Controls
# ----------------------------
horizon_label = st.selectbox("Forecast horizon", ["24 hours", "7 days"])
horizon = 24 if horizon_label == "24 hours" else 168

show_band = st.checkbox("Show uncertainty band (±1.96 * residual std)", value=True)

# ----------------------------
# Load assets
# ----------------------------
model, model_err = load_model()
df, df_err = load_default_data()

if model_err:
    st.error(model_err)
    st.info("Fix: set the correct MODEL_PATH in app/utils.py to your real model filename in /models.")
    st.stop()

if df_err:
    st.error(df_err)
    st.stop()

# ----------------------------
# Debug info (hidden)
# ----------------------------
expected = get_expected_features(model)
with st.expander("Debug: model expected features"):
    st.write(expected)

# ----------------------------
# Infer columns
# ----------------------------
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

# ----------------------------
# Prepare time series
# ----------------------------
df_ts = ensure_datetime_index(df, dt_col)
y = df_ts[y_col].astype(float).dropna()

# ----------------------------
# Backtest forecast
# ----------------------------
result, err = backtest_forecast(model, y, horizon)
if err:
    st.error(err)
    st.stop()

pred_index, y_true, y_pred = result

# Backtest window caption (what window is being evaluated)
st.caption(
    f"Backtest window used: {pred_index[0]} to {pred_index[-1]} "
    f"(most recent {horizon_label.lower()})."
)

# ----------------------------
# Metrics
# ----------------------------
mae_v = mae(y_true, y_pred)
rmse_v = rmse(y_true, y_pred)
mape_v = mape(y_true, y_pred)

c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mae_v:.3f}")
c2.metric("RMSE", f"{rmse_v:.3f}")
c3.metric("MAPE (%)", f"{mape_v:.2f}")

# ----------------------------
# Uncertainty band (based on residual std)
# ----------------------------
residuals = (y_true - y_pred)
res_std = float(pd.Series(residuals).dropna().std(ddof=0))

upper = y_pred + 1.96 * res_std
lower = y_pred - 1.96 * res_std

# ----------------------------
# Plot (compact + readable)
# ----------------------------
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(pred_index, y_true, label="Actual", linewidth=2)
ax.plot(pred_index, y_pred, label="Forecast", linewidth=2)

if show_band:
    ax.fill_between(pred_index, lower, upper, alpha=0.15, label="Uncertainty band")

ax.set_title(f"Backtest Forecast ({horizon_label})")
ax.set_ylabel(y_col)
ax.set_xlabel("Time")

# Fix overlapping x-axis labels + margins
fig.autofmt_xdate(rotation=30)
ax.margins(x=0)
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
st.pyplot(fig, clear_figure=True)

# ----------------------------
# Forecast table + download
# ----------------------------
out = pd.DataFrame({
    "datetime": pred_index,
    "actual": y_true,
    "forecast": y_pred,
    "residual": residuals
})

csv = out.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download forecast table (CSV)",
    data=csv,
    file_name=f"forecast_backtest_{horizon}h.csv",
    mime="text/csv"
)

with st.expander("See forecast table"):
    st.dataframe(out, use_container_width=True)
