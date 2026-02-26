import streamlit as st

from utils import (
    load_model,
    load_threshold_config,
    load_default_data,
    get_threshold_value,
    infer_datetime_col,
    infer_target_col,
)

# ----------------------------
# Page config (safe)
# ----------------------------
st.set_page_config(
    page_title="Energy Forecasting & Anomaly Detection",
    page_icon="⚡",
    layout="wide",
)

# ----------------------------
# Header
# ----------------------------
st.title("Energy Consumption Forecasting & Anomaly Detection")
st.caption(
    "Forecast hourly energy usage and flag unusual deviations using residual-based anomaly detection."
)

# ----------------------------
# Load system assets (safe read-only)
# ----------------------------
model, model_err = load_model()
cfg, cfg_err = load_threshold_config()
df, df_err = load_default_data()

# ----------------------------
# Sidebar: System status
# ----------------------------
st.sidebar.header("System Status")

# Model
if model_err:
    st.sidebar.error("Model not loaded")
else:
    st.sidebar.success("Model loaded")

# Threshold
if cfg_err:
    st.sidebar.error("Threshold config not loaded")
else:
    thr = get_threshold_value(cfg)
    if thr is None:
        st.sidebar.warning("Threshold config loaded (value not parsed)")
    else:
        st.sidebar.success(f"Threshold loaded (thr={thr:.4f})")

# Data
if df_err:
    st.sidebar.error("Default data not loaded")
else:
    st.sidebar.success(f"Default data loaded ({df.shape[0]:,} rows, {df.shape[1]} cols)")

st.sidebar.divider()
st.sidebar.caption("Tip: If a page shows an error, check model path, threshold JSON keys, and required columns.")

# ----------------------------
# Main: Quick overview cards
# ----------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Model status", "Ready" if not model_err else "Missing")

with c2:
    if cfg_err:
        st.metric("Threshold", "Missing")
    else:
        thr = get_threshold_value(cfg)
        st.metric("Threshold", f"{thr:.4f}" if thr is not None else "Loaded")

with c3:
    st.metric("Dataset", "Ready" if not df_err else "Missing")

with c4:
    if not df_err:
        dt_col = infer_datetime_col(df)
        y_col = infer_target_col(df)
        st.metric("Detected columns", f"{dt_col or 'N/A'} | {y_col or 'N/A'}")
    else:
        st.metric("Detected columns", "N/A")

st.divider()

# ----------------------------
# What the app does (professional)
# ----------------------------
st.subheader("What you can do here")

left, right = st.columns([1.2, 1])

with left:
    st.markdown(
        """
**This dashboard supports two workflows:**

1. **Forecasting**  
   Generate a short-horizon forecast (24 hours / 7 days) and evaluate performance on a backtest window.

2. **Anomaly detection (residual-based)**  
   Flag time points where the model's prediction error is unusually large, using a percentile-based threshold.

Use the navigation on the left to explore each module.
        """.strip()
    )

with right:
    st.markdown(
        """
**Pages**
- **Overview**: trend + peak hour + weekly average + top-load alerts
- **Forecast**: backtest forecast, metrics, and uncertainty band
- **Anomalies**: highlighted anomaly points + severity table + download
- **Upload**: upload CSV, validate, score, and download report
        """.strip()
    )

st.divider()

# ----------------------------
# Getting started section
# ----------------------------
st.subheader("Getting started")

if model_err or cfg_err or df_err:
    st.warning("Some system components are not ready. Fix the items below, then refresh.")
    if model_err:
        st.write(f"- Model error: {model_err}")
    if cfg_err:
        st.write(f"- Threshold config error: {cfg_err}")
    if df_err:
        st.write(f"- Default data error: {df_err}")
    st.stop()

st.success("System ready. Choose a page from the sidebar to begin.")

with st.expander("Recommended usage"):
    st.markdown(
        """
- Start with **Overview** to understand trends and peak usage.
- Go to **Forecast** to check 24h vs 7d performance.
- Use **Anomalies** to inspect unusual deviations.
- Use **Upload** to score your own CSV and download a report.
        """.strip()
    )

with st.expander("Upload format requirements"):
    st.markdown(
        """
Your uploaded CSV should contain:
- A datetime column (default name: `datetime`)
- A target column (default name: `Global_active_power`)

If your CSV has different names, you can change them on the Upload page before scoring.
        """.strip()
    )