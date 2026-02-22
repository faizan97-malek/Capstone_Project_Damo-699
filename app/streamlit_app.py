import sys
from pathlib import Path

# --- ensure project root is on path ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.inference import predict
from src.simulator import generate_sensor_state
from src.shap_explain import get_top_shap_drivers
from src.ttf_proxy import estimate_ttf_proxy

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(
    page_title="Real-Time Predictive Maintenance Dashboard",
    layout="wide"
)

st.title("Real-Time Predictive Maintenance Dashboard")

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("Controls")

refresh_seconds = st.sidebar.slider(
    "Refresh interval (seconds)",
    1, 10, 2
)

top_k = st.sidebar.slider(
    "Top SHAP drivers to show",
    3, 10, 5
)

# --------------------------------------------------
# Session history
# --------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------
# Simulate live sensor
# --------------------------------------------------
sensor = generate_sensor_state()

# ✅ SAFE Product ID handling
product_id = sensor.get("Product ID", "SIM-" + pd.Timestamp.now().strftime("%H%M%S"))

# --------------------------------------------------
# Prediction
# --------------------------------------------------
result = predict(sensor)
risk_prob = result.get("risk_probability", 0.0)

# --------------------------------------------------
# SHAP drivers (best effort)
# --------------------------------------------------
try:
    shap_drivers = get_top_shap_drivers(sensor, top_k=top_k)
except Exception as e:
    shap_drivers = []
    st.warning(f"SHAP drivers not available yet: {e}")

# --------------------------------------------------
# Save to history
# --------------------------------------------------
st.session_state.history.append(
    {
        "ts": pd.Timestamp.now(),
        "risk_probability": risk_prob,
    }
)

# --------------------------------------------------
# TTF Proxy
# --------------------------------------------------
ttf_info = estimate_ttf_proxy(
    history=st.session_state.history,
    current_sensor=sensor
)
ttf_value = ttf_info.get("ttf_proxy_min")

# --------------------------------------------------
# KPI ROW (NOW WITH PRODUCT ID)
# --------------------------------------------------
c0, c1, c2, c3, c4 = st.columns(5)

c0.metric("Product ID", product_id)
c1.metric("Risk Probability", f"{risk_prob:.2%}")
c2.metric("Risk Level", result.get("risk_label", "N/A"))
c3.metric("Top Drivers Count", str(len(shap_drivers)))

if ttf_value is None:
    c4.metric("TTF Proxy (min)", "Calculating…")
else:
    c4.metric("TTF Proxy (min)", f"{ttf_value}")

st.caption(
    "TTF Proxy is an analytical estimate based on session trend and tool wear "
    "(AI4I dataset does not provide true time-to-failure)."
)

st.divider()

# ==================================================
# Centered Gauge
# ==================================================
g_left, g_mid, g_right = st.columns([1, 2, 1])

with g_mid:
    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_prob * 100,
            title={"text": "Failure Risk Gauge (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.25},
                "steps": [
                    {"range": [0, 30], "color": "#2ecc71"},
                    {"range": [30, 70], "color": "#f1c40f"},
                    {"range": [70, 100], "color": "#e74c3c"},
                ],
            },
        )
    )

    gauge_fig.update_layout(height=320)
    st.plotly_chart(gauge_fig, use_container_width=True)

st.divider()

# --------------------------------------------------
# Sensor + SHAP panels
# --------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Current Sensor State")

    sensor_df = pd.DataFrame(
        list(sensor.items()),
        columns=["feature", "value"]
    )

    def _pretty(v):
        if isinstance(v, (int, float)):
            return round(v, 3)
        return v

    sensor_df["value"] = sensor_df["value"].apply(_pretty)

    st.dataframe(sensor_df, use_container_width=True, hide_index=True)

with right:
    st.subheader("Top SHAP Drivers")
    if shap_drivers:
        st.dataframe(
            pd.DataFrame(shap_drivers),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No SHAP drivers to display yet.")

st.divider()

# --------------------------------------------------
# Risk trend
# --------------------------------------------------
st.subheader("Risk Trend (this session)")
hist_df = pd.DataFrame(st.session_state.history)
st.line_chart(hist_df.set_index("ts")[["risk_probability"]])

# --------------------------------------------------
# Auto refresh
# --------------------------------------------------
time.sleep(refresh_seconds)
st.rerun()