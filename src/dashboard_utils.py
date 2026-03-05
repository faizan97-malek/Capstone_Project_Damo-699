from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.inference import predict, compute_ttf_proxy
from src.shap_explain import get_top_shap_drivers
from src.simulator import step_sensor_state

ROOT = Path(__file__).resolve().parents[1]

# Matplotlib dark theme helpers (KM/Cox plots)
def apply_dark_mpl(ax, fig=None):
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

# Preferred project location
DATA_PATH = ROOT / "data" / "cleaned" / "ai4i2020_cleaned.csv"

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
    # hide internal simulator fields (those starting with "_")
    clean_sensor = {
        k: v for k, v in sensor.items()
        if not k.startswith("_")
    }

    df = pd.DataFrame(
        [{"Feature": k, "Value": v} for k, v in clean_sensor.items()]
    )
    return df

def make_risk_gauge(prob: float, threshold: float = 0.18):
    value = float(np.clip(prob * 100.0, 0.0, 100.0))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Failure Risk Gauge (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0,  35], "color": "#2ecc71"},
                    {"range": [35, 70], "color": "#f1c40f"},
                    {"range": [70, 100], "color": "#e74c3c"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.85,
                    "value": threshold * 100,
                },
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

# Global "Aging" Simulation
def _init_sim_state():
    if "machines" not in st.session_state:
        st.session_state.machines = {}  # pid -> {"state": dict, "last_tick": int}
    if "sim_tick" not in st.session_state:
        st.session_state.sim_tick = 0
    if "sim_running" not in st.session_state:
        st.session_state.sim_running = False
    if "history" not in st.session_state:
        st.session_state.history = []
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
    if "drift_by_id" not in st.session_state:
        st.session_state.drift_by_id = {}  # pid -> machine-specific drift params

def _attach_sim_internals(sensor: dict) -> dict:
    # Keep your existing keys, just add internal state for fluctuation
    s = dict(sensor)

    # simulator internals (used by step_sensor_state)
    s.setdefault("_t", 0)
    s.setdefault("_air_target", float(s["Air temperature [K]"]))
    s.setdefault("_rpm_base", float(s["Rotational speed [rpm]"]))
    s.setdefault("_workload", 0.5)
    s.setdefault("_last_shock", 0.0)
    return s

def _get_machine_params(product_id: str, base_sensor: dict, rng: np.random.Generator) -> dict:
    if product_id in st.session_state.drift_by_id:
        return st.session_state.drift_by_id[product_id]

    mtype = str(base_sensor.get("Type", "M")).upper()

    if mtype == "H":
        wear_range = (0.3, 0.9)
        torque_mu, torque_sigma = 0.04, 0.18
        air_mu, air_sigma = 0.01, 0.06
        proc_mu, proc_sigma = 0.02, 0.08
        rpm_mu, rpm_sigma = -0.5, 5.0
    elif mtype == "L":
        wear_range = (0.1, 0.5)
        torque_mu, torque_sigma = 0.02, 0.10
        air_mu, air_sigma = 0.01, 0.04
        proc_mu, proc_sigma = 0.01, 0.05
        rpm_mu, rpm_sigma = -0.3, 3.0
    else:
        wear_range = (0.2, 0.7)
        torque_mu, torque_sigma = 0.03, 0.14
        air_mu, air_sigma = 0.01, 0.05
        proc_mu, proc_sigma = 0.01, 0.06
        rpm_mu, rpm_sigma = -0.4, 4.0

    params = {
        "wear_per_tick":    float(rng.uniform(*wear_range)),
        "air_mu":           float(rng.normal(air_mu, 0.005)),
        "air_sigma":        float(rng.uniform(air_sigma * 0.7, air_sigma * 1.1)),
        "proc_mu":          float(rng.normal(proc_mu, 0.005)),
        "proc_sigma":       float(rng.uniform(proc_sigma * 0.7, proc_sigma * 1.2)),
        "torque_mu":        float(rng.normal(torque_mu, 0.01)),
        "torque_sigma":     float(rng.uniform(torque_sigma * 0.7, torque_sigma * 1.2)),
        "rpm_mu":           float(rng.normal(rpm_mu, 0.2)),
        "rpm_sigma":        float(rng.uniform(rpm_sigma * 0.7, rpm_sigma * 1.2)),
        "wear_heat_gain":   float(rng.uniform(0.0002, 0.0008)),
        "wear_torque_gain": float(rng.uniform(0.001, 0.003)),
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

    # Clip to training data realistic range (p2–p98) so simulator never
    # generates out-of-distribution inputs that guarantee 100% risk
    s["Air temperature [K]"]     = float(np.clip(s["Air temperature [K]"],     296.0, 304.0))
    s["Process temperature [K]"] = float(np.clip(s["Process temperature [K]"], 307.0, 313.5))
    s["Torque [Nm]"]             = float(np.clip(s["Torque [Nm]"],              10.0,  63.0))
    s["Rotational speed [rpm]"]  = float(np.clip(s["Rotational speed [rpm]"],  1200.0, 2100.0))
    s["Tool wear [min]"]         = float(np.clip(s["Tool wear [min]"],            0.0,  260.0))

    return s

def get_or_create_machine_state(product_id: str, base_sensor: dict, step: bool = True) -> dict:
    """
    Returns the simulated machine state for a given product_id.
    - Creates state from base_sensor once.
    - Advances it only when sim_tick increases (and step=True).
    """
    machines = st.session_state.machines
    tick_now = int(st.session_state.sim_tick)

    if product_id not in machines:
        init_state = _attach_sim_internals(base_sensor)
        machines[product_id] = {"state": init_state, "last_tick": tick_now}
        return init_state

    record = machines[product_id]
    state = record["state"]
    last_tick = int(record.get("last_tick", tick_now))

    if not step:
        return state

    # advance as many ticks as needed (usually 0 or 1)
    steps = max(0, tick_now - last_tick)
    for _ in range(steps):
        state = step_sensor_state(state)

    record["state"] = state
    record["last_tick"] = tick_now
    machines[product_id] = record

    return state

def reset_simulation():
    st.session_state.machines = {}
    st.session_state.sim_tick = 0
    st.session_state.history = []

# Shared (Common) KPI + Gauge block
def render_common_kpis_and_gauge(sensor: dict, top_k: int):
    model_input = {k: v for k, v in sensor.items() if k != "Product ID"}
    result = predict(model_input)

    risk_prob = float(result.get("risk_probability", 0.0))
    risk_label = str(result.get("risk_label", "N/A"))
    ttf_info = compute_ttf_proxy(model_input)
    ttf_value = float(ttf_info.get("ttf_min", 0.0))
    ttf_method = str(ttf_info.get("method", "unknown"))

    shap_drivers = []
    try:
        shap_drivers = get_top_shap_drivers(model_input, top_k=top_k)
    except Exception as e:
        st.warning(f"SHAP drivers not available: {e}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Product ID", sensor.get("Product ID", "N/A"))
    k2.metric("Risk Probability", f"{risk_prob:.2%}")
    k3.metric("Risk Level", risk_label)
    k4.metric("TTF (min)", f"{ttf_value:.1f}", help=f"Method: {ttf_method}")
    if ttf_method == "wear_rule_fallback":
        k4.caption("⚠️ Fallback estimate")
    else:
        k4.caption("✅ Regression model")

    threshold = float(result.get("threshold_used", 0.18))

    g1, g2, g3 = st.columns([1, 2, 1])
    with g2:
        st.plotly_chart(make_risk_gauge(risk_prob, threshold=threshold), use_container_width=True)

    return risk_prob, shap_drivers

# Survival caching helpers (used by Page 3)
def _df_fingerprint(df: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(df, index=False).values
    return f"{int(h.sum())}_{len(df)}_{len(df.columns)}"


@st.cache_resource(show_spinner=False)
def _fit_cox_cached(df_cox: pd.DataFrame, _fp: str):
    # Import lazily so Page 1/2 don't require lifelines.
    from src.survival_analysis import fit_cox_model

    return fit_cox_model(df_cox)