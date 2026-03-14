from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


MAINTENANCE_TRIGGER_PROB = 0.90
MAINTENANCE_TICKS_BY_TYPE = {
    "H": 20,
    "M": 15,
    "L": 12,
}

ROOT = Path(__file__).resolve().parents[1]


# We created these helper functions because the KM and Cox plots on page 3
# use matplotlib which defaults to a white background. Since our dashboard
# uses Streamlit dark mode, we need to override the colors manually so the
# charts dont look out of place.

def apply_dark_mpl(ax, fig=None):
    if fig is not None:
        fig.patch.set_alpha(0.0)
        fig.patch.set_facecolor((0, 0, 0, 0))

    ax.set_facecolor((0, 0, 0, 0))

    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.35)

    ax.grid(True, alpha=0.18)

    leg = ax.get_legend()
    if leg is not None:
        frame = leg.get_frame()
        frame.set_alpha(0.25)
        frame.set_edgecolor("white")
        for t in leg.get_texts():
            t.set_color("white")


def finalize_fig(fig):
    # We standardize the figure size here because the KM and Cox plots
    # were rendering at different heights which made the layout uneven.
    fig.set_size_inches(6.0, 4.0)
    fig.tight_layout(pad=1.0)
    return fig


# We define the dataset path here so every function in this module
# reads from the same location. The fallback paths exist because
# team members have different folder structures on their machines.
DATA_PATH = ROOT / "data" / "cleaned" / "ai4i2020_cleaned.csv"

FALLBACK_PATHS = [
    ROOT / "ai4i2020_cleaned.csv",
    Path("/mnt/data/ai4i2020_cleaned.csv"),
]

# We exclude these columns from the page 2 editor because they are
# either the target variable or failure mode flags that would cause
# data leakage if an operator edited them.
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

    # We handle this rename because some older versions of the cleaned
    # CSV used UDI as the identifier column instead of Product ID.
    if "Product ID" not in df.columns and "UDI" in df.columns:
        df = df.rename(columns={"UDI": "Product ID"})

    df["Product ID"] = df["Product ID"].astype(str)
    return df


def build_sensor_from_row(row: pd.Series) -> dict:
    # We extract only the columns that the model expects as input,
    # plus Product ID for display purposes in the dashboard.
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
    # We filter out keys starting with underscore because those are
    # internal simulator fields that would confuse the operator.
    clean_sensor = {
        k: v for k, v in sensor.items()
        if not k.startswith("_")
    }

    df = pd.DataFrame(
        [{"Feature": k, "Value": v} for k, v in clean_sensor.items()]
    )
    return df


def make_risk_gauge(prob: float, threshold: float = 0.18):
    # We chose 35/70 as the gauge color boundaries because they divide
    # the 0-100% range into three visually meaningful zones. The white
    # threshold line shows where the models decision boundary sits.
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


# We initialize all session state keys in one place so the app doesnt
# crash with KeyError on the first load. Each key has a sensible default.
def _init_sim_state():
    if "machines" not in st.session_state:
        st.session_state.machines = {}
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
        st.session_state.drift_by_id = {}
    # Number of machine slots chosen by the slider (1-10)
    if "sim_num_machines" not in st.session_state:
        st.session_state.sim_num_machines = 1
    # List of product IDs chosen via the per-slot dropdowns
    if "sim_selected_pids" not in st.session_state:
        st.session_state.sim_selected_pids = []


def _attach_sim_internals(sensor: dict) -> dict:
    # We add hidden fields because the simulator needs to track
    # internal state between ticks, but we dont want them shown
    # in the sensor table.
    s = dict(sensor)

    s.setdefault("_t", 0)
    s.setdefault("_air_target", float(s["Air temperature [K]"]))
    s.setdefault("_rpm_base", float(s["Rotational speed [rpm]"]))
    s.setdefault("_workload", 0.5)
    s.setdefault("_last_shock", 0.0)

    # We store the original sensor snapshot so maintenance can
    # restore the machine back to its pre-simulation baseline.
    s.setdefault("_base_Air temperature [K]", float(s["Air temperature [K]"]))
    s.setdefault("_base_Process temperature [K]", float(s["Process temperature [K]"]))
    s.setdefault("_base_Rotational speed [rpm]", float(s["Rotational speed [rpm]"]))
    s.setdefault("_base_Torque [Nm]", float(s["Torque [Nm]"]))
    s.setdefault("_base_Tool wear [min]", float(s["Tool wear [min]"]))

    # Maintenance state
    s.setdefault("_maintenance_active", False)
    s.setdefault("_maintenance_ticks_left", 0)

    return s

def _get_maintenance_duration(sensor: dict) -> int:
    mtype = str(sensor.get("Type", "M")).upper()
    return int(MAINTENANCE_TICKS_BY_TYPE.get(mtype, 15))


def _start_maintenance(sensor: dict) -> dict:
    sensor["_maintenance_active"] = True
    sensor["_maintenance_ticks_left"] = _get_maintenance_duration(sensor)
    return sensor


@st.cache_data(show_spinner=False)
def _get_fleet_medians() -> dict:
    """
    Compute fleet-wide medians for Torque and Rotational speed once and
    cache them. These are used as the post-repair reset targets so every
    machine returns to a healthy, neutral operating point rather than its
    original (possibly already-degraded) baseline values.
    """
    df = load_cleaned_dataset()
    return {
        "Torque [Nm]":            float(df["Torque [Nm]"].median()),
        "Rotational speed [rpm]": float(df["Rotational speed [rpm]"].median()),
    }


def _finish_maintenance(sensor: dict) -> dict:
    # After repair we restore temperatures to the machine's own baseline
    # (the physical environment hasn't changed) but reset Torque and RPM
    # to fleet medians — representing a freshly serviced, neutral state —
    # and zero out Tool wear to reflect new tooling.
    medians = _get_fleet_medians()

    sensor["Air temperature [K]"]     = float(sensor.get("_base_Air temperature [K]",     sensor["Air temperature [K]"]))
    sensor["Process temperature [K]"] = float(sensor.get("_base_Process temperature [K]", sensor["Process temperature [K]"]))
    sensor["Rotational speed [rpm]"]  = medians["Rotational speed [rpm]"]
    sensor["Torque [Nm]"]             = medians["Torque [Nm]"]
    sensor["Tool wear [min]"]         = 0.0

    # Reset simulator internals so the machine resumes cleanly from the new state
    sensor["_air_target"] = float(sensor["Air temperature [K]"])
    sensor["_rpm_base"]   = medians["Rotational speed [rpm]"]
    sensor["_workload"]   = 0.5
    sensor["_last_shock"] = 0.0

    sensor["_maintenance_active"]    = False
    sensor["_maintenance_ticks_left"] = 0
    return sensor


def _advance_with_maintenance(sensor: dict) -> dict:
    # If already under maintenance, just count ticks down
    if bool(sensor.get("_maintenance_active", False)):
        ticks_left = int(sensor.get("_maintenance_ticks_left", 0)) - 1
        sensor["_maintenance_ticks_left"] = max(0, ticks_left)

        if sensor["_maintenance_ticks_left"] <= 0:
            sensor = _finish_maintenance(sensor)

        sensor["_t"] = int(sensor.get("_t", 0)) + 1
        return sensor

    # Normal simulation tick
    from src.simulator import step_sensor_state  # lazy – avoids circular import
    sensor = step_sensor_state(sensor)

    # Check if machine should enter maintenance after this tick
    model_input = {
        "Type": sensor.get("Type", "M"),
        "Air temperature [K]": float(sensor.get("Air temperature [K]", 0)),
        "Process temperature [K]": float(sensor.get("Process temperature [K]", 0)),
        "Rotational speed [rpm]": float(sensor.get("Rotational speed [rpm]", 0)),
        "Torque [Nm]": float(sensor.get("Torque [Nm]", 0)),
        "Tool wear [min]": float(sensor.get("Tool wear [min]", 0)),
    }

    try:
        from src.inference import predict  # lazy – avoids circular import
        result = predict(model_input)
        risk_prob = float(result.get("risk_probability", 0.0))
        if risk_prob >= MAINTENANCE_TRIGGER_PROB:
            sensor = _start_maintenance(sensor)
    except Exception:
        pass

    return sensor

def _get_machine_params(product_id: str, base_sensor: dict, rng: np.random.Generator) -> dict:
    # We cache drift parameters per machine so each product ID has
    # consistent degradation behavior across the entire simulation.
    # Without this, the same machine would drift differently each tick.
    if product_id in st.session_state.drift_by_id:
        return st.session_state.drift_by_id[product_id]

    mtype = str(base_sensor.get("Type", "M")).upper()

    # We use different drift ranges per type because Type H machines
    # are high-quality and degrade faster under stress, while Type L
    # machines are low-quality with gentler operating conditions.
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

    # We add wear-dependent heating because in real machines, worn tools
    # generate more friction which raises both air and process temperatures.
    wear_heat = wear * params["wear_heat_gain"]
    s["Air temperature [K]"] = float(s["Air temperature [K]"] + air_drift + (0.3 * wear_heat))
    s["Process temperature [K]"] = float(s["Process temperature [K]"] + air_drift + proc_extra + wear_heat)

    torque_drift = float(rng.normal(params["torque_mu"], params["torque_sigma"]))
    wear_torque = wear * params["wear_torque_gain"]
    s["Torque [Nm]"] = float(s["Torque [Nm]"] + torque_drift + wear_torque)

    rpm_drift = float(rng.normal(params["rpm_mu"], params["rpm_sigma"]))
    s["Rotational speed [rpm]"] = float(s["Rotational speed [rpm]"] + rpm_drift)

    # We clip all values to the training datas realistic range (roughly p2 to p98)
    # because if the simulator drifts outside what the model has seen, it would
    # always predict 100% risk which makes the dashboard useless.
    s["Air temperature [K]"]     = float(np.clip(s["Air temperature [K]"],     296.0, 304.0))
    s["Process temperature [K]"] = float(np.clip(s["Process temperature [K]"], 307.0, 313.5))
    s["Torque [Nm]"]             = float(np.clip(s["Torque [Nm]"],              10.0,  63.0))
    s["Rotational speed [rpm]"]  = float(np.clip(s["Rotational speed [rpm]"],  1200.0, 2100.0))
    s["Tool wear [min]"]         = float(np.clip(s["Tool wear [min]"],            0.0,  260.0))

    return s


def get_or_create_machine_state(product_id: str, base_sensor: dict, step: bool = True) -> dict:
    # We track each machines state separately so when the user switches
    # between product IDs, each machine remembers where it left off
    # instead of resetting to its original dataset values.
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

    # We advance by however many ticks were missed since this machine
    # was last updated. Usually this is 0 or 1 but it can be more if
    # the user was viewing a different machine for several ticks.
    steps = max(0, tick_now - last_tick)
    for _ in range(steps):
        state = _advance_with_maintenance(state)

    record["state"] = state
    record["last_tick"] = tick_now
    machines[product_id] = record

    return state


def reset_simulation():
    st.session_state.machines = {}
    st.session_state.sim_tick = 0
    st.session_state.history = []


def step_all_machines(df_source: pd.DataFrame):
    # We only advance the machines the user has explicitly selected via
    # the sidebar dropdowns. Stepping the full 10 000-row dataset every
    # tick was the primary cause of slow refreshes — now we touch at
    # most 10 rows regardless of dataset size.
    tick_now = int(st.session_state.sim_tick)
    machines = st.session_state.machines

    selected_pids = [str(p) for p in st.session_state.get("sim_selected_pids", [])]
    if not selected_pids:
        return

    # Build an index once so look-ups are O(1) instead of a full scan.
    # We keep a separate dict of rows with Product ID restored because
    # set_index() removes "Product ID" as a column, which makes
    # build_sensor_from_row raise a KeyError when it tries row["Product ID"].
    df_indexed = df_source.set_index("Product ID")

    for pid in selected_pids:
        if pid not in df_indexed.index:
            continue

        if pid not in machines:
            # .loc[pid] returns a Series if unique, DataFrame if duplicates
            raw = df_indexed.loc[pid]
            if isinstance(raw, pd.DataFrame):
                raw = raw.iloc[0]
            row = raw.copy()
            row["Product ID"] = pid   # restore the column removed by set_index
            base = build_sensor_from_row(row)
            init_state = _attach_sim_internals(base)
            machines[pid] = {"state": init_state, "last_tick": tick_now}
            continue

        record = machines[pid]
        state = record["state"]
        last_tick = int(record.get("last_tick", tick_now))

        steps = max(0, tick_now - last_tick)
        for _ in range(steps):
            state = _advance_with_maintenance(state)

        record["state"] = state
        record["last_tick"] = tick_now
        machines[pid] = record


def render_common_kpis_and_gauge(sensor: dict, top_k: int):
    # Lazy imports here break the circular dependency chain:
    # dashboard_utils → inference → (indirectly) dashboard_utils
    from src.inference import predict, compute_ttf_proxy  # noqa: PLC0415
    from src.shap_explain import get_top_shap_drivers      # noqa: PLC0415
    # We run prediction and TTF estimation here because all three
    # pages need to display the same KPI cards and gauge.
    model_input = {
        k: v for k, v in sensor.items()
        if k != "Product ID" and not str(k).startswith("_")
    }

    maintenance_active = bool(sensor.get("_maintenance_active", False))
    operational_status = "Under Maintenance" if maintenance_active else "Operational"

    if maintenance_active:
        risk_prob = None
        risk_label = "N/A"

        ttf_info = compute_ttf_proxy(model_input)
        ttf_value = float(ttf_info.get("ttf_min", 0.0))
        ttf_method = str(ttf_info.get("method", "unknown"))

        shap_drivers = []
    else:
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

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Product ID", sensor.get("Product ID", "N/A"))
    k2.metric("Risk Probability", "N/A" if maintenance_active else f"{risk_prob:.2%}")
    k3.metric("Risk Level", "N/A" if maintenance_active else risk_label)
    k4.metric("TTF (min)", "N/A" if maintenance_active else f"{ttf_value:.1f}", help=f"Method: {ttf_method}")
    k5.metric("Operational Status", operational_status)

    if ttf_method == "wear_rule_fallback":
        k4.caption("Fallback estimate")
    else:
        k4.caption("Regression model")

    g1, g2, g3 = st.columns([1, 2, 1])
    with g2:
        if maintenance_active:
            st.markdown(
                """
                <div style="
                    height:320px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    border:1px solid rgba(255,255,255,0.15);
                    border-radius:12px;
                    background-color:rgba(255,255,255,0.02);
                    font-size:28px;
                    font-weight:600;
                    color:white;
                ">
                    Failure Risk Gauge: N/A
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            threshold = float(result.get("threshold_used", 0.18))
            st.plotly_chart(make_risk_gauge(risk_prob, threshold=threshold), use_container_width=True)

    return (0.0 if risk_prob is None else risk_prob), shap_drivers


def _df_fingerprint(df: pd.DataFrame) -> str:
    # We hash the dataframe so Streamlits cache knows when the Cox
    # model needs to be refit. Without this, it would either refit
    # every refresh (slow) or never update after page 2 edits.
    h = pd.util.hash_pandas_object(df, index=False).values
    return f"{int(h.sum())}_{len(df)}_{len(df.columns)}"


@st.cache_resource(show_spinner=False)
def _fit_cox_cached(df_cox: pd.DataFrame, _fp: str):
    # We import lifelines lazily here so pages 1 and 2 dont crash
    # if lifelines is not installed on the users machine.
    from src.survival_analysis import fit_cox_model
    return fit_cox_model(df_cox)