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


# -------------------------------------------------
# Gauge helper (NEW — professor will like this)
# -------------------------------------------------
def plot_risk_gauge(probability: float):
    """
    Create a visual gauge for failure probability.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": "Failure Risk (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.25},
                "steps": [
                    {"range": [0, 40], "color": "#2ecc71"},
                    {"range": [40, 70], "color": "#f1c40f"},
                    {"range": [70, 100], "color": "#e74c3c"},
                ],
            },
        )
    )

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# -------------------------------------------------
# Streamlit page config
# -------------------------------------------------
st.set_page_config(
    page_title="Real-Time Predictive Maintenance Dashboard",
    layout="wide",
)

st.title("Real-Time Predictive Maintenance Dashboard")


# -------------------------------------------------
# Sidebar controls
# -------------------------------------------------
st.sidebar.header("Controls")
refresh_seconds = st.sidebar.slider("Refresh interval (seconds)", 1, 10, 2)
top_k = st.sidebar.slider("Top SHAP drivers to show", 3, 10, 5)


# -------------------------------------------------
# Session history (persists during session)
# -------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# -------------------------------------------------
# Simulated live sensor snapshot
# -------------------------------------------------
sensor = generate_sensor_state()


# -------------------------------------------------
# Prediction
# -------------------------------------------------
result = predict(sensor)
risk_prob = result.get("risk_probability", 0.0)
risk_label = result.get("risk_label", "N/A")


# -------------------------------------------------
# SHAP drivers (best-effort)
# -------------------------------------------------
try:
    shap_drivers = get_top_shap_drivers(sensor, top_k=top_k)
except Exception as e:
    shap_drivers = []
    st.warning(f"SHAP drivers not available yet: {e}")


# -------------------------------------------------
# Save to history
# -------------------------------------------------
st.session_state.history.append(
    {
        "ts": pd.Timestamp.now(),
        "risk_probability": risk_prob,
    }
)


# -------------------------------------------------
# KPIs
# -------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Risk Probability", f"{risk_prob:.2%}")
c2.metric("Risk Level", risk_label)
c3.metric("Top Drivers Count", str(len(shap_drivers)))

st.divider()


# -------------------------------------------------
# Main layout
# -------------------------------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Current Sensor State")
    st.json(sensor)

with right:
    st.subheader("Failure Risk Gauge")
    gauge_fig = plot_risk_gauge(risk_prob)
    st.plotly_chart(gauge_fig, use_container_width=True)

    st.subheader("Top SHAP Drivers")
    if shap_drivers:
        st.dataframe(pd.DataFrame(shap_drivers), use_container_width=True)
    else:
        st.info("No SHAP drivers to display yet.")


st.divider()


# -------------------------------------------------
# Risk trend chart
# -------------------------------------------------
st.subheader("Risk Trend (this session)")
hist_df = pd.DataFrame(st.session_state.history)
st.line_chart(hist_df.set_index("ts")[["risk_probability"]])


# -------------------------------------------------
# Auto refresh loop
# -------------------------------------------------
time.sleep(refresh_seconds)
st.rerun()