# app/streamlit_app.py
import sys
from pathlib import Path

# --- ensure project root is on path (so `src.*` imports work when running streamlit) ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from src.inference import predict, compute_ttf_proxy
from src.shap_explain import get_top_shap_drivers


# ---------------------------
# Data
# ---------------------------
DATA_PATH = ROOT / "data" / "cleaned" / "ai4i2020_cleaned.csv"

# Columns you DO NOT want in Page 2 table
FAILURE_COLS = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]


@st.cache_data(show_spinner=False)
def load_cleaned_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find cleaned dataset at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Some versions use "UDI" instead of "Product ID"
    if "Product ID" not in df.columns and "UDI" in df.columns:
        df = df.rename(columns={"UDI": "Product ID"})

    df["Product ID"] = df["Product ID"].astype(str)
    return df


def build_sensor_from_row(row: pd.Series) -> dict:
    """Minimal snapshot expected by model + Product ID for UI."""
    return {
        "Product ID": str(row["Product ID"]),
        "Type": str(row["Type"]),
        "Air temperature [K]": float(row["Air temperature [K]"]),
        "Process temperature [K]": float(row["Process temperature [K]"]),
        "Rotational speed [rpm]": float(row["Rotational speed [rpm]"]),
        "Torque [Nm]": float(row["Torque [Nm]"]),
        "Tool wear [min]": float(row["Tool wear [min]"]),
    }


def sensor_table(sensor: dict) -> pd.DataFrame:
    return pd.DataFrame([{"feature": k, "value": v} for k, v in sensor.items()])


def make_risk_gauge(prob: float):
    value = float(np.clip(prob * 100.0, 0.0, 100.0))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": "Failure Risk Gauge (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 35], "color": "#2ecc71"},
                    {"range": [35, 70], "color": "#f1c40f"},
                    {"range": [70, 100], "color": "#e74c3c"},
                ],
            },
        )
    )
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def make_trend_chart(hist_df: pd.DataFrame, title: str):
    if hist_df.empty:
        return None

    hist_df = hist_df.sort_values("ts")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist_df["ts"],
            y=hist_df["risk_probability"],
            mode="lines+markers",
            name="Risk Probability",
        )
    )

    fig.update_layout(
        title=title,
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Time",
        yaxis_title="Risk Probability",
        yaxis=dict(range=[0, 1]),
    )
    return fig


# ---------------------------
# Global "Aging" Simulation
# ---------------------------
def _init_sim_state():
    """
    State:
      - sim_tick: global tick
      - base_by_id / drift_by_id / last_tick_by_id / current_by_id: simulation cache
      - sim_running: start/pause
      - df_edit: committed editable dataset used by both pages
      - df_edit_draft: table draft edits (Page 2) waiting for Done
      - page1_pid: Product ID selector for page 1 (drives KPIs on page 1)
      - page2_pid: Product ID selector for page 2 (drives KPIs on page 2)
      - history: for trends
    """
    if "sim_tick" not in st.session_state:
        st.session_state.sim_tick = 0

    if "base_by_id" not in st.session_state:
        st.session_state.base_by_id = {}

    if "drift_by_id" not in st.session_state:
        st.session_state.drift_by_id = {}

    if "last_tick_by_id" not in st.session_state:
        st.session_state.last_tick_by_id = {}

    if "current_by_id" not in st.session_state:
        st.session_state.current_by_id = {}

    if "rng" not in st.session_state:
        st.session_state.rng = np.random.default_rng(42)

    if "sim_running" not in st.session_state:
        st.session_state.sim_running = False

    if "df_edit" not in st.session_state:
        st.session_state.df_edit = None

    if "df_edit_draft" not in st.session_state:
        st.session_state.df_edit_draft = None

    if "page1_pid" not in st.session_state:
        st.session_state.page1_pid = None

    if "page2_pid" not in st.session_state:
        st.session_state.page2_pid = None

    if "history" not in st.session_state:
        st.session_state.history = []


def _get_machine_params(product_id: str, base_sensor: dict, rng: np.random.Generator) -> dict:
    if product_id in st.session_state.drift_by_id:
        return st.session_state.drift_by_id[product_id]

    mtype = str(base_sensor.get("Type", "M")).upper()

    if mtype == "H":
        wear_range = (1.2, 3.0)
        torque_mu, torque_sigma = 0.25, 0.55
        air_mu, air_sigma = 0.08, 0.20
        proc_mu, proc_sigma = 0.18, 0.30
        rpm_mu, rpm_sigma = -2.0, 12.0
    elif mtype == "L":
        wear_range = (0.4, 1.6)
        torque_mu, torque_sigma = 0.10, 0.35
        air_mu, air_sigma = 0.03, 0.14
        proc_mu, proc_sigma = 0.08, 0.18
        rpm_mu, rpm_sigma = -1.0, 8.0
    else:
        wear_range = (0.7, 2.2)
        torque_mu, torque_sigma = 0.16, 0.45
        air_mu, air_sigma = 0.05, 0.16
        proc_mu, proc_sigma = 0.12, 0.22
        rpm_mu, rpm_sigma = -1.5, 10.0

    params = {
        "wear_per_tick": float(rng.uniform(*wear_range)),
        "air_mu": float(rng.normal(air_mu, 0.02)),
        "air_sigma": float(rng.uniform(air_sigma * 0.7, air_sigma * 1.1)),
        "proc_mu": float(rng.normal(proc_mu, 0.04)),
        "proc_sigma": float(rng.uniform(proc_sigma * 0.7, proc_sigma * 1.2)),
        "torque_mu": float(rng.normal(torque_mu, 0.06)),
        "torque_sigma": float(rng.uniform(torque_sigma * 0.7, torque_sigma * 1.2)),
        "rpm_mu": float(rng.normal(rpm_mu, 0.6)),
        "rpm_sigma": float(rng.uniform(rpm_sigma * 0.7, rpm_sigma * 1.2)),
        "wear_heat_gain": float(rng.uniform(0.002, 0.006)),
        "wear_torque_gain": float(rng.uniform(0.01, 0.03)),
    }

    st.session_state.drift_by_id[product_id] = params
    return params


def _apply_one_tick(sensor: dict, params: dict, rng: np.random.Generator) -> dict:
    s = dict(sensor)

    s["Tool wear [min]"] = float(s["Tool wear [min]"] + params["wear_per_tick"])
    wear = float(s["Tool wear [min]"])

    air_drift = float(rng.normal(params["air_mu"], params["air_sigma"]))
    proc_extra = float(rng.normal(params["proc_mu"], params["proc_sigma"]))

    wear_heat = wear * params["wear_heat_gain"]
    s["Air temperature [K]"] = float(s["Air temperature [K]"] + air_drift + (0.3 * wear_heat))
    s["Process temperature [K]"] = float(s["Process temperature [K]"] + air_drift + proc_extra + wear_heat)

    torque_drift = float(rng.normal(params["torque_mu"], params["torque_sigma"]))
    wear_torque = wear * params["wear_torque_gain"]
    s["Torque [Nm]"] = float(s["Torque [Nm]"] + torque_drift + wear_torque)

    rpm_drift = float(rng.normal(params["rpm_mu"], params["rpm_sigma"]))
    s["Rotational speed [rpm]"] = float(s["Rotational speed [rpm]"] + rpm_drift)

    s["Air temperature [K]"] = float(np.clip(s["Air temperature [K]"], 290, 320))
    s["Process temperature [K]"] = float(np.clip(s["Process temperature [K]"], 295, 335))
    s["Torque [Nm]"] = float(np.clip(s["Torque [Nm]"], 0, 140))
    s["Rotational speed [rpm]"] = float(np.clip(s["Rotational speed [rpm]"], 0, 3500))
    s["Tool wear [min]"] = float(np.clip(s["Tool wear [min]"], 0, 300))

    return s


def get_or_create_machine_state(product_id: str, base_sensor: dict) -> dict:
    rng = st.session_state.rng

    if product_id not in st.session_state.base_by_id:
        st.session_state.base_by_id[product_id] = dict(base_sensor)

    if product_id not in st.session_state.current_by_id:
        st.session_state.current_by_id[product_id] = dict(base_sensor)

    if product_id not in st.session_state.last_tick_by_id:
        st.session_state.last_tick_by_id[product_id] = st.session_state.sim_tick

    params = _get_machine_params(product_id, st.session_state.base_by_id[product_id], rng)

    last_tick = st.session_state.last_tick_by_id[product_id]
    now_tick = st.session_state.sim_tick
    steps = max(now_tick - last_tick, 0)

    cur = st.session_state.current_by_id[product_id]
    for _ in range(steps):
        cur = _apply_one_tick(cur, params, rng)

    st.session_state.current_by_id[product_id] = cur
    st.session_state.last_tick_by_id[product_id] = now_tick

    return cur


def reset_simulation():
    st.session_state.sim_tick = 0
    st.session_state.drift_by_id = {}
    st.session_state.last_tick_by_id = {}
    st.session_state.current_by_id = {}
    st.session_state.history = []


# ---------------------------
# Shared (Common) KPI + Gauge block
# ---------------------------
def render_common_kpis_and_gauge(sensor: dict, top_k: int):
    model_input = {k: v for k, v in sensor.items() if k != "Product ID"}
    result = predict(model_input)

    risk_prob = float(result.get("risk_probability", 0.0))
    risk_label = str(result.get("risk_label", "N/A"))
    ttf_wear = compute_ttf_proxy(model_input)

    shap_drivers = []
    try:
        shap_drivers = get_top_shap_drivers(model_input, top_k=top_k)
    except Exception as e:
        st.warning(f"SHAP drivers not available: {e}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Product ID", sensor.get("Product ID", "N/A"))
    k2.metric("Risk Probability", f"{risk_prob:.2%}")
    k3.metric("Risk Level", risk_label)
    k4.metric("TTF Proxy (Wear Proxy, min)", f"{ttf_wear:.1f}")

    g1, g2, g3 = st.columns([1, 2, 1])
    with g2:
        st.plotly_chart(make_risk_gauge(risk_prob), use_container_width=True)

    return risk_prob, shap_drivers


# ---------------------------
# App
# ---------------------------
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("Predictive Maintenance Dashboard")

df_all = load_cleaned_dataset()
_init_sim_state()

# committed dataset used everywhere
if st.session_state.df_edit is None:
    st.session_state.df_edit = df_all.copy()

df_source = st.session_state.df_edit

# product ids
product_ids = df_source["Product ID"].unique().tolist()
product_ids.sort()

# default independent selectors
if st.session_state.page1_pid is None and product_ids:
    st.session_state.page1_pid = product_ids[0]
if st.session_state.page2_pid is None and product_ids:
    st.session_state.page2_pid = product_ids[0]

# Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["1) Simulation Dashboard", "2) Editable Data Table"], index=0)

# Common SHAP control (fine to keep common)
st.sidebar.markdown("---")
top_k = st.sidebar.slider("Top SHAP drivers to show", 3, 10, 5)

# Page 1 controls
if page == "1) Simulation Dashboard":
    st.sidebar.markdown("---")
    st.sidebar.header("Simulation Controls")

    # ONE dropdown only (drives KPIs + page 1 "Select Product ID" content)
    st.session_state.page1_pid = st.sidebar.selectbox(
        "Page 1 Product ID (drives KPIs here)",
        product_ids,
        index=product_ids.index(st.session_state.page1_pid) if st.session_state.page1_pid in product_ids else 0,
        key="page1_pid_selectbox",
    )

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("▶️ Start", use_container_width=True):
            st.session_state.sim_running = True
            st.rerun()
    with c2:
        if st.button("⏸️ Pause", use_container_width=True):
            st.session_state.sim_running = False
            st.rerun()

    st.sidebar.caption(
        f"Simulation status: {'🟢 Running' if st.session_state.sim_running else '🔴 Paused'}"
    )

    refresh_seconds = st.sidebar.slider("Refresh interval (seconds)", 1, 10, 2)
    simulate_degradation = st.sidebar.toggle("Simulate degradation (aging)", value=True)
    auto_refresh = st.sidebar.toggle("Auto-refresh", value=True)

    if st.sidebar.button("Reset simulation (all machines)"):
        reset_simulation()
        st.success("Simulation reset. All machines are back to original dataset values.")
        st.rerun()

# Page 2 controls
else:
    st.sidebar.markdown("---")
    st.sidebar.header("Table Controls")

    # independent Product ID selector for page 2 (drives KPIs on page 2)
    st.session_state.page2_pid = st.sidebar.selectbox(
        "Page 2 Product ID (drives KPIs here)",
        product_ids,
        index=product_ids.index(st.session_state.page2_pid) if st.session_state.page2_pid in product_ids else 0,
        key="page2_pid_selectbox",
    )

    simulate_degradation = False
    auto_refresh = False
    refresh_seconds = 2
    # keep sim from rerunning while table editing
    st.session_state.sim_running = False


# ---------------------------
# Build the sensor that drives the KPIs & gauge (depends on page)
# ---------------------------
active_pid = st.session_state.page1_pid if page == "1) Simulation Dashboard" else st.session_state.page2_pid
kpi_row = df_source[df_source["Product ID"] == str(active_pid)].iloc[0]
kpi_base_sensor = build_sensor_from_row(kpi_row)

if page == "1) Simulation Dashboard" and simulate_degradation:
    # advance tick only when running + auto refresh
    if st.session_state.sim_running and auto_refresh:
        st.session_state.sim_tick += 1
    kpi_sensor = get_or_create_machine_state(str(active_pid), kpi_base_sensor)
else:
    kpi_sensor = kpi_base_sensor


# ---------------------------
# COMMON KPIs + Gauge (shown on BOTH pages)
# ---------------------------
risk_prob, shap_drivers = render_common_kpis_and_gauge(kpi_sensor, top_k=top_k)

st.caption(
    "Page 1: optional aging simulation. Page 2: edit the dataset and press Done to apply changes."
)

st.divider()


# ---------------------------
# PAGE 1: Simulation Dashboard
# ---------------------------
if page == "1) Simulation Dashboard":
    st.subheader("Simulation Dashboard")

    mode = st.sidebar.radio(
        "Evaluation mode (page content)",
        ["Random Live (dataset sampling)", "Select Product ID (dataset snapshot)"],
        index=1,
        key="page1_mode",
    )

    if mode == "Random Live (dataset sampling)":
        row = df_source.sample(1).iloc[0]
        base_sensor = build_sensor_from_row(row)
        sensor = base_sensor

        if simulate_degradation and st.session_state.sim_running and auto_refresh:
            sensor = get_or_create_machine_state(sensor["Product ID"], base_sensor)

    else:
        # IMPORTANT FIX: use the SAME dropdown (page1_pid). No second dropdown.
        page_content_pid = st.session_state.page1_pid
        row = df_source[df_source["Product ID"] == str(page_content_pid)].iloc[0]
        base_sensor = build_sensor_from_row(row)
        sensor = get_or_create_machine_state(str(page_content_pid), base_sensor) if simulate_degradation else base_sensor

    # history (trend uses page-content machine)
    st.session_state.history.append(
        {
            "ts": pd.Timestamp.now(),
            "product_id": sensor.get("Product ID", "N/A"),
            "risk_probability": float(risk_prob),
            "tick": st.session_state.sim_tick,
        }
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Current Sensor State (Page 1 Content)")
        st.dataframe(sensor_table(sensor), use_container_width=True)

    with right:
        st.subheader("Top SHAP Drivers (Common)")
        if shap_drivers:
            st.dataframe(pd.DataFrame(shap_drivers), use_container_width=True)
        else:
            st.info("No SHAP drivers to display yet.")

    st.divider()

    st.subheader("Risk Trend (Page 1 Content Product ID)")
    hist = pd.DataFrame(st.session_state.history)
    pid_now = sensor.get("Product ID", "N/A")
    hist_pid = hist[hist["product_id"] == pid_now].copy()
    trend_fig = make_trend_chart(hist_pid, title=f"Risk Probability Trend — {pid_now}")

    if trend_fig is None:
        st.info("No trend data yet for this Product ID.")
    else:
        st.plotly_chart(trend_fig, use_container_width=True)

    # Auto refresh loop (only when running)
    if auto_refresh and st.session_state.sim_running:
        time.sleep(refresh_seconds)
        st.rerun()


# ---------------------------
# PAGE 2: Editable Data Table with DONE button
# ---------------------------
else:
    st.subheader("Editable Data Table (Excel-like)")

    st.info(
        "Edit values in the table below. Changes are **not applied** until you press **✅ Done**."
    )

    # Use draft if exists; else start from committed df_source
    if st.session_state.df_edit_draft is None:
        st.session_state.df_edit_draft = df_source.copy()

    df_draft = st.session_state.df_edit_draft

    # Build view without failure cols
    cols_to_drop = [c for c in FAILURE_COLS if c in df_draft.columns]
    df_view = df_draft.drop(columns=cols_to_drop, errors="ignore")

    # Keep Product ID / Type at left
    preferred_left = [c for c in ["Product ID", "Type"] if c in df_view.columns]
    other_cols = [c for c in df_view.columns if c not in preferred_left]
    df_view = df_view[preferred_left + other_cols]

    edited_df_view = st.data_editor(
        df_view,
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        key="data_editor_all",
    )

    # Update draft (but do NOT commit yet)
    df_updated_draft = df_draft.copy()
    for col in edited_df_view.columns:
        df_updated_draft[col] = edited_df_view[col]
    st.session_state.df_edit_draft = df_updated_draft

    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("✅ Done", use_container_width=True):
            # Commit edits
            st.session_state.df_edit = st.session_state.df_edit_draft.copy()
            st.success("Edits applied. KPIs & gauge updated.")
            st.rerun()
    with c2:
        st.caption("Press **Done** to apply edits and refresh KPIs/gauge.")

    st.divider()
    st.caption(
        "Tip: Page 2 KPIs are driven by **Page 2 Product ID** selector in the sidebar (independent from Page 1)."
    )