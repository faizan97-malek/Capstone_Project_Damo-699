"""Utility functions used by the Streamlit dashboard.

Important: This module is a pure refactor of logic that previously lived in
`app/streamlit_app.py`. The intent is to keep the dashboard output identical.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.inference import predict, compute_ttf_proxy
from src.shap_explain import get_top_shap_drivers

# NOTE: Survival helpers (_df_fingerprint / _fit_cox_cached) live here so
# app/streamlit_app.py stays clean. These functions are copied from the
# original streamlit_app.py with NO behavior changes.

# Project root (src/ -> project root)
ROOT = Path(__file__).resolve().parents[1]

# ---------------------------
# Matplotlib dark theme helpers (KM/Cox plots)
# ---------------------------
def apply_dark_mpl(ax, fig=None):
    """
    Makes matplotlib plots look good on Streamlit dark background.
    """
    if fig is not None:
        fig.patch.set_alpha(0.0)  # transparent figure background
        fig.patch.set_facecolor((0, 0, 0, 0))

    # transparent axes background
    ax.set_facecolor((0, 0, 0, 0))

    # light text
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    # light spines
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.35)

    # subtle grid
    ax.grid(True, alpha=0.18)

    # legend
    leg = ax.get_legend()
    if leg is not None:
        frame = leg.get_frame()
        frame.set_alpha(0.25)
        frame.set_edgecolor("white")
        for t in leg.get_texts():
            t.set_color("white")


def finalize_fig(fig):
    """
    Standardizes sizing/padding so KM and Cox align.
    """
    fig.set_size_inches(6.0, 4.0)  # same for both plots
    fig.tight_layout(pad=1.0)
    return fig


# ---------------------------
# Data
# ---------------------------
# Preferred project location
DATA_PATH = ROOT / "data" / "cleaned" / "ai4i2020_cleaned.csv"

# Fallbacks (in case you run directly from a sandbox / different structure)
FALLBACK_PATHS = [
    ROOT / "ai4i2020_cleaned.csv",
    Path("/mnt/data/ai4i2020_cleaned.csv"),
]

# Columns you DO NOT want in Page 2 table
FAILURE_COLS = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]


@st.cache_data(show_spinner=False)
def load_cleaned_dataset() -> pd.DataFrame:
    path = DATA_PATH
    if not path.exists():
        for fp in FALLBACK_PATHS:
            if fp.exists():
                path = fp
                break

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find cleaned dataset.\nTried:\n- {DATA_PATH}\n"
            + "\n".join([f"- {p}" for p in FALLBACK_PATHS])
        )

    df = pd.read_csv(path)

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
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
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
        height=300,
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
      - df_edit: committed editable dataset used by all pages
      - df_edit_draft: table draft edits (Page 2) waiting for Done
      - page1_pid: Product ID selector for page 1 (drives KPIs on page 1)
      - page2_pid: Product ID selector for page 2 (drives KPIs on page 2)
      - page3_pid: Product ID selector for page 3 (drives KPIs + survival on page 3)  ✅ independent
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

    if "page3_pid" not in st.session_state:
        st.session_state.page3_pid = None

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
# Survival caching helpers (used by Page 3)
# ---------------------------
def _df_fingerprint(df: pd.DataFrame) -> str:
    """Fingerprint a dataframe for caching.

    Copied from the original dashboard code (no behavior change).
    """
    h = pd.util.hash_pandas_object(df, index=False).values
    return f"{int(h.sum())}_{len(df)}_{len(df.columns)}"


@st.cache_resource(show_spinner=False)
def _fit_cox_cached(df_cox: pd.DataFrame, _fp: str):
    """Fit Cox model with Streamlit resource caching.

    `_fp` is included to invalidate cache when df_cox changes.
    Behavior matches the original dashboard.
    """
    # Import lazily so Page 1/2 don't require lifelines.
    from src.survival_analysis import fit_cox_model

    return fit_cox_model(df_cox)

