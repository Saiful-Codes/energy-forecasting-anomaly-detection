import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    load_default_data,
    infer_datetime_col,
    infer_target_col,
    ensure_datetime_index,
    top_percent_threshold,
)

st.title("1) Overview")

# ----------------------------
# Load data
# ----------------------------
df, err = load_default_data()
if err:
    st.error(err)
    st.stop()

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

df_ts = ensure_datetime_index(df, dt_col)
y = df_ts[y_col].astype(float).dropna()

# ----------------------------
# Controls
# ----------------------------
st.subheader("Trend")
view = st.selectbox("Time range", ["Last 7 days", "Last 30 days", "Last 90 days", "All data"])

if view == "Last 7 days":
    y_view = y.last("7D")
elif view == "Last 30 days":
    y_view = y.last("30D")
elif view == "Last 90 days":
    y_view = y.last("90D")
else:
    y_view = y

# ----------------------------
# Trend chart (compact + readable)
# ----------------------------
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_view.index, y_view.values, linewidth=1.5)
ax.set_title(f"Energy consumption trend ({view})")
ax.set_xlabel("Time")
ax.set_ylabel(y_col)
fig.autofmt_xdate(rotation=30)
ax.margins(x=0)
ax.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig, clear_figure=True)

# ----------------------------
# Peak hour summary
# ----------------------------
st.subheader("Peak hour summary")

# Create hour-of-day profile
hourly_profile = y.groupby(y.index.hour).mean().sort_index()
peak_hour = int(hourly_profile.idxmax())
peak_val = float(hourly_profile.max())

# Quick metrics
c1, c2, c3 = st.columns(3)
c1.metric("Peak hour (avg)", f"{peak_hour:02d}:00")
c2.metric("Avg usage at peak hour", f"{peak_val:.3f}")
c3.metric("Overall average usage", f"{float(y.mean()):.3f}")

# Show profile as a simple line
fig2, ax2 = plt.subplots(figsize=(10, 3.5))
ax2.plot(hourly_profile.index, hourly_profile.values, linewidth=2)
ax2.set_title("Average usage by hour of day")
ax2.set_xlabel("Hour of day")
ax2.set_ylabel(y_col)
ax2.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig2, clear_figure=True)

# ----------------------------
# Weekly average usage
# ----------------------------
st.subheader("Weekly average usage")

weekly_avg = y.resample("W").mean()
weekly_change = (weekly_avg.iloc[-1] - weekly_avg.iloc[-2]) if len(weekly_avg) >= 2 else np.nan
weekly_pct = (weekly_change / weekly_avg.iloc[-2] * 100.0) if len(weekly_avg) >= 2 and weekly_avg.iloc[-2] != 0 else np.nan

c4, c5 = st.columns(2)
c4.metric("Most recent weekly average", f"{float(weekly_avg.iloc[-1]):.3f}" if len(weekly_avg) else "N/A")
c5.metric(
    "Weekly change (vs previous)",
    f"{float(weekly_change):.3f} ({float(weekly_pct):.1f}%)" if not np.isnan(weekly_pct) else "N/A"
)

fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(weekly_avg.index, weekly_avg.values, linewidth=2)
ax3.set_title("Weekly average energy usage")
ax3.set_xlabel("Week")
ax3.set_ylabel(y_col)
fig3.autofmt_xdate(rotation=30)
ax3.margins(x=0)
ax3.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig3, clear_figure=True)

# Download weekly averages
weekly_df = weekly_avg.reset_index()
weekly_df.columns = ["week", "weekly_avg_usage"]
csv_weekly = weekly_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download weekly averages (CSV)",
    data=csv_weekly,
    file_name="weekly_average_usage.csv",
    mime="text/csv"
)

# ----------------------------
# Top 5% load alert
# ----------------------------
st.subheader("Top 5% load alert")

top5_thr = top_percent_threshold(y, pct=5.0)
top5_points = y[y >= top5_thr]

c6, c7, c8 = st.columns(3)
c6.metric("Top 5% threshold", f"{top5_thr:.3f}")
c7.metric("Count of top 5% hours", f"{int(top5_points.shape[0]):,}")
c8.metric("Max observed usage", f"{float(y.max()):.3f}")

if len(top5_points) > 0:
    st.warning(
        f"Alert: {len(top5_points):,} hours are in the top 5% of usage "
        f"(>= {top5_thr:.3f}). Review these for potential high-load events."
    )

# Show only most recent top load points (keep table compact)
show_n = st.slider("Show most recent top-load rows", min_value=10, max_value=200, value=50, step=10)
top_table = top5_points.sort_index(ascending=False).head(show_n).reset_index()
top_table.columns = ["datetime", "usage"]

st.dataframe(top_table, use_container_width=True)

csv_top = top_table.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download top-load rows (CSV)",
    data=csv_top,
    file_name="top5_load_rows.csv",
    mime="text/csv"
)