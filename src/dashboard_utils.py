# from __future__ import annotations
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# import streamlit as st


# MAINTENANCE_TRIGGER_PROB = 0.90
# MAINTENANCE_TICKS_BY_TYPE = {
#     "H": 20,
#     "M": 15,
#     "L": 12,
# }

# ROOT = Path(__file__).resolve().parents[1]


# # We created these helper functions because the KM and Cox plots on page 3
# # use matplotlib which defaults to a white background. Since our dashboard
# # uses Streamlit dark mode, we need to override the colors manually so the
# # charts dont look out of place.

# def apply_dark_mpl(ax, fig=None):
#     if fig is not None:
#         fig.patch.set_alpha(0.0)
#         fig.patch.set_facecolor((0, 0, 0, 0))

#     ax.set_facecolor((0, 0, 0, 0))

#     ax.title.set_color("white")
#     ax.xaxis.label.set_color("white")
#     ax.yaxis.label.set_color("white")

#     ax.tick_params(axis="x", colors="white")
#     ax.tick_params(axis="y", colors="white")

#     for spine in ax.spines.values():
#         spine.set_color("white")
#         spine.set_alpha(0.35)

#     ax.grid(True, alpha=0.18)

#     leg = ax.get_legend()
#     if leg is not None:
#         frame = leg.get_frame()
#         frame.set_alpha(0.25)
#         frame.set_edgecolor("white")
#         for t in leg.get_texts():
#             t.set_color("white")


# def finalize_fig(fig):
#     # We standardize the figure size here because the KM and Cox plots
#     # were rendering at different heights which made the layout uneven.
#     fig.set_size_inches(6.0, 4.0)
#     fig.tight_layout(pad=1.0)
#     return fig


# # We define the dataset path here so every function in this module
# # reads from the same location. The fallback paths exist because
# # team members have different folder structures on their machines.
# DATA_PATH = ROOT / "data" / "cleaned" / "ai4i2020_cleaned.csv"

# FALLBACK_PATHS = [
#     ROOT / "ai4i2020_cleaned.csv",
#     Path("/mnt/data/ai4i2020_cleaned.csv"),
# ]

# # We exclude these columns from the page 2 editor because they are
# # either the target variable or failure mode flags that would cause
# # data leakage if an operator edited them.
# FAILURE_COLS = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]


# @st.cache_data(show_spinner=False)
# def load_cleaned_dataset() -> pd.DataFrame:
#     path = DATA_PATH
#     if not path.exists():
#         for fp in FALLBACK_PATHS:
#             if fp.exists():
#                 path = fp
#                 break

#     if not path.exists():
#         raise FileNotFoundError(
#             f"Could not find cleaned dataset.\nTried:\n- {DATA_PATH}\n"
#             + "\n".join([f"- {p}" for p in FALLBACK_PATHS])
#         )

#     df = pd.read_csv(path)

#     # We handle this rename because some older versions of the cleaned
#     # CSV used UDI as the identifier column instead of Product ID.
#     if "Product ID" not in df.columns and "UDI" in df.columns:
#         df = df.rename(columns={"UDI": "Product ID"})

#     df["Product ID"] = df["Product ID"].astype(str)
#     return df


# def build_sensor_from_row(row: pd.Series) -> dict:
#     # We extract only the columns that the model expects as input,
#     # plus Product ID for display purposes in the dashboard.
#     return {
#         "Product ID": str(row["Product ID"]),
#         "Type": str(row["Type"]),
#         "Air temperature [K]": float(row["Air temperature [K]"]),
#         "Process temperature [K]": float(row["Process temperature [K]"]),
#         "Rotational speed [rpm]": float(row["Rotational speed [rpm]"]),
#         "Torque [Nm]": float(row["Torque [Nm]"]),
#         "Tool wear [min]": float(row["Tool wear [min]"]),
#     }


# def sensor_table(sensor: dict) -> pd.DataFrame:
#     # We filter out keys starting with underscore because those are
#     # internal simulator fields that would confuse the operator.
#     clean_sensor = {
#         k: v for k, v in sensor.items()
#         if not k.startswith("_")
#     }

#     df = pd.DataFrame(
#         [{"Feature": k, "Value": v} for k, v in clean_sensor.items()]
#     )
#     return df


# def make_risk_gauge(prob: float, threshold: float = 0.18):
#     # We chose 35/70 as the gauge color boundaries because they divide
#     # the 0-100% range into three visually meaningful zones. The white
#     # threshold line shows where the models decision boundary sits.
#     value = float(np.clip(prob * 100.0, 0.0, 100.0))
#     fig = go.Figure(
#         go.Indicator(
#             mode="gauge+number",
#             value=value,
#             number={"suffix": "%", "valueformat": ".1f"},
#             title={"text": "Failure Risk Gauge (%)"},
#             gauge={
#                 "axis": {"range": [0, 100]},
#                 "steps": [
#                     {"range": [0,  35], "color": "#2ecc71"},
#                     {"range": [35, 70], "color": "#f1c40f"},
#                     {"range": [70, 100], "color": "#e74c3c"},
#                 ],
#                 "threshold": {
#                     "line": {"color": "white", "width": 3},
#                     "thickness": 0.85,
#                     "value": threshold * 100,
#                 },
#             },
#         )
#     )
#     fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
#     return fig


# def make_trend_chart(hist_df: pd.DataFrame, title: str):
#     if hist_df.empty:
#         return None

#     hist_df = hist_df.sort_values("ts")

#     fig = go.Figure()
#     fig.add_trace(
#         go.Scatter(
#             x=hist_df["ts"],
#             y=hist_df["risk_probability"],
#             mode="lines+markers",
#             name="Risk Probability",
#         )
#     )

#     fig.update_layout(
#         title=title,
#         height=300,
#         margin=dict(l=10, r=10, t=40, b=10),
#         xaxis_title="Time",
#         yaxis_title="Risk Probability",
#         yaxis=dict(range=[0, 1]),
#     )
#     return fig


# # We initialize all session state keys in one place so the app doesnt
# # crash with KeyError on the first load. Each key has a sensible default.
# def _init_sim_state():
#     if "machines" not in st.session_state:
#         st.session_state.machines = {}
#     if "sim_tick" not in st.session_state:
#         st.session_state.sim_tick = 0
#     if "sim_running" not in st.session_state:
#         st.session_state.sim_running = False
#     if "history" not in st.session_state:
#         st.session_state.history = []
#     if "df_edit" not in st.session_state:
#         st.session_state.df_edit = None
#     if "df_edit_draft" not in st.session_state:
#         st.session_state.df_edit_draft = None
#     if "page1_pid" not in st.session_state:
#         st.session_state.page1_pid = None
#     if "page2_pid" not in st.session_state:
#         st.session_state.page2_pid = None
#     if "page3_pid" not in st.session_state:
#         st.session_state.page3_pid = None
#     if "drift_by_id" not in st.session_state:
#         st.session_state.drift_by_id = {}
#     # Number of machine slots chosen by the slider (1-10)
#     if "sim_num_machines" not in st.session_state:
#         st.session_state.sim_num_machines = 1
#     # List of product IDs chosen via the per-slot dropdowns
#     if "sim_selected_pids" not in st.session_state:
#         st.session_state.sim_selected_pids = []


# def _attach_sim_internals(sensor: dict) -> dict:
#     # We add hidden fields because the simulator needs to track
#     # internal state between ticks, but we dont want them shown
#     # in the sensor table.
#     s = dict(sensor)

#     s.setdefault("_t", 0)
#     s.setdefault("_air_target", float(s["Air temperature [K]"]))
#     s.setdefault("_rpm_base", float(s["Rotational speed [rpm]"]))
#     s.setdefault("_workload", 0.5)
#     s.setdefault("_last_shock", 0.0)

#     # We store the original sensor snapshot so maintenance can
#     # restore the machine back to its pre-simulation baseline.
#     s.setdefault("_base_Air temperature [K]", float(s["Air temperature [K]"]))
#     s.setdefault("_base_Process temperature [K]", float(s["Process temperature [K]"]))
#     s.setdefault("_base_Rotational speed [rpm]", float(s["Rotational speed [rpm]"]))
#     s.setdefault("_base_Torque [Nm]", float(s["Torque [Nm]"]))
#     s.setdefault("_base_Tool wear [min]", float(s["Tool wear [min]"]))

#     # Maintenance state
#     s.setdefault("_maintenance_active", False)
#     s.setdefault("_maintenance_ticks_left", 0)

#     return s

# def _get_maintenance_duration(sensor: dict) -> int:
#     mtype = str(sensor.get("Type", "M")).upper()
#     return int(MAINTENANCE_TICKS_BY_TYPE.get(mtype, 15))


# def _start_maintenance(sensor: dict) -> dict:
#     sensor["_maintenance_active"] = True
#     sensor["_maintenance_ticks_left"] = _get_maintenance_duration(sensor)
#     return sensor


# @st.cache_data(show_spinner=False)
# def _get_fleet_medians() -> dict:
#     """
#     Compute fleet-wide medians for Torque and Rotational speed once and
#     cache them. These are used as the post-repair reset targets so every
#     machine returns to a healthy, neutral operating point rather than its
#     original (possibly already-degraded) baseline values.
#     """
#     df = load_cleaned_dataset()
#     return {
#         "Torque [Nm]":            float(df["Torque [Nm]"].median()),
#         "Rotational speed [rpm]": float(df["Rotational speed [rpm]"].median()),
#     }


# def _finish_maintenance(sensor: dict) -> dict:
#     # After repair we restore temperatures to the machine's own baseline
#     # (the physical environment hasn't changed) but reset Torque and RPM
#     # to fleet medians — representing a freshly serviced, neutral state —
#     # and zero out Tool wear to reflect new tooling.
#     medians = _get_fleet_medians()

#     sensor["Air temperature [K]"]     = float(sensor.get("_base_Air temperature [K]",     sensor["Air temperature [K]"]))
#     sensor["Process temperature [K]"] = float(sensor.get("_base_Process temperature [K]", sensor["Process temperature [K]"]))
#     sensor["Rotational speed [rpm]"]  = medians["Rotational speed [rpm]"]
#     sensor["Torque [Nm]"]             = medians["Torque [Nm]"]
#     sensor["Tool wear [min]"]         = 0.0

#     # Reset simulator internals so the machine resumes cleanly from the new state
#     sensor["_air_target"] = float(sensor["Air temperature [K]"])
#     sensor["_rpm_base"]   = medians["Rotational speed [rpm]"]
#     sensor["_workload"]   = 0.5
#     sensor["_last_shock"] = 0.0

#     sensor["_maintenance_active"]    = False
#     sensor["_maintenance_ticks_left"] = 0
#     return sensor


# def _advance_with_maintenance(sensor: dict) -> dict:
#     # If already under maintenance, just count ticks down
#     if bool(sensor.get("_maintenance_active", False)):
#         ticks_left = int(sensor.get("_maintenance_ticks_left", 0)) - 1
#         sensor["_maintenance_ticks_left"] = max(0, ticks_left)

#         if sensor["_maintenance_ticks_left"] <= 0:
#             sensor = _finish_maintenance(sensor)

#         sensor["_t"] = int(sensor.get("_t", 0)) + 1
#         return sensor

#     # Normal simulation tick
#     from src.simulator import step_sensor_state  # lazy – avoids circular import
#     sensor = step_sensor_state(sensor)

#     # Check if machine should enter maintenance after this tick
#     model_input = {
#         "Type": sensor.get("Type", "M"),
#         "Air temperature [K]": float(sensor.get("Air temperature [K]", 0)),
#         "Process temperature [K]": float(sensor.get("Process temperature [K]", 0)),
#         "Rotational speed [rpm]": float(sensor.get("Rotational speed [rpm]", 0)),
#         "Torque [Nm]": float(sensor.get("Torque [Nm]", 0)),
#         "Tool wear [min]": float(sensor.get("Tool wear [min]", 0)),
#     }

#     try:
#         from src.inference import predict  # lazy – avoids circular import
#         result = predict(model_input)
#         risk_prob = float(result.get("risk_probability", 0.0))
#         if risk_prob >= MAINTENANCE_TRIGGER_PROB:
#             sensor = _start_maintenance(sensor)
#     except Exception:
#         pass

#     return sensor

# def _get_machine_params(product_id: str, base_sensor: dict, rng: np.random.Generator) -> dict:
#     # We cache drift parameters per machine so each product ID has
#     # consistent degradation behavior across the entire simulation.
#     # Without this, the same machine would drift differently each tick.
#     if product_id in st.session_state.drift_by_id:
#         return st.session_state.drift_by_id[product_id]

#     mtype = str(base_sensor.get("Type", "M")).upper()

#     # We use different drift ranges per type because Type H machines
#     # are high-quality and degrade faster under stress, while Type L
#     # machines are low-quality with gentler operating conditions.
#     if mtype == "H":
#         wear_range = (0.3, 0.9)
#         torque_mu, torque_sigma = 0.04, 0.18
#         air_mu, air_sigma = 0.01, 0.06
#         proc_mu, proc_sigma = 0.02, 0.08
#         rpm_mu, rpm_sigma = -0.5, 5.0
#     elif mtype == "L":
#         wear_range = (0.1, 0.5)
#         torque_mu, torque_sigma = 0.02, 0.10
#         air_mu, air_sigma = 0.01, 0.04
#         proc_mu, proc_sigma = 0.01, 0.05
#         rpm_mu, rpm_sigma = -0.3, 3.0
#     else:
#         wear_range = (0.2, 0.7)
#         torque_mu, torque_sigma = 0.03, 0.14
#         air_mu, air_sigma = 0.01, 0.05
#         proc_mu, proc_sigma = 0.01, 0.06
#         rpm_mu, rpm_sigma = -0.4, 4.0

#     params = {
#         "wear_per_tick":    float(rng.uniform(*wear_range)),
#         "air_mu":           float(rng.normal(air_mu, 0.005)),
#         "air_sigma":        float(rng.uniform(air_sigma * 0.7, air_sigma * 1.1)),
#         "proc_mu":          float(rng.normal(proc_mu, 0.005)),
#         "proc_sigma":       float(rng.uniform(proc_sigma * 0.7, proc_sigma * 1.2)),
#         "torque_mu":        float(rng.normal(torque_mu, 0.01)),
#         "torque_sigma":     float(rng.uniform(torque_sigma * 0.7, torque_sigma * 1.2)),
#         "rpm_mu":           float(rng.normal(rpm_mu, 0.2)),
#         "rpm_sigma":        float(rng.uniform(rpm_sigma * 0.7, rpm_sigma * 1.2)),
#         "wear_heat_gain":   float(rng.uniform(0.0002, 0.0008)),
#         "wear_torque_gain": float(rng.uniform(0.001, 0.003)),
#     }

#     st.session_state.drift_by_id[product_id] = params
#     return params


# def _apply_one_tick(sensor: dict, params: dict, rng: np.random.Generator) -> dict:
#     s = dict(sensor)

#     s["Tool wear [min]"] = float(s["Tool wear [min]"] + params["wear_per_tick"])
#     wear = float(s["Tool wear [min]"])

#     air_drift = float(rng.normal(params["air_mu"], params["air_sigma"]))
#     proc_extra = float(rng.normal(params["proc_mu"], params["proc_sigma"]))

#     # We add wear-dependent heating because in real machines, worn tools
#     # generate more friction which raises both air and process temperatures.
#     wear_heat = wear * params["wear_heat_gain"]
#     s["Air temperature [K]"] = float(s["Air temperature [K]"] + air_drift + (0.3 * wear_heat))
#     s["Process temperature [K]"] = float(s["Process temperature [K]"] + air_drift + proc_extra + wear_heat)

#     torque_drift = float(rng.normal(params["torque_mu"], params["torque_sigma"]))
#     wear_torque = wear * params["wear_torque_gain"]
#     s["Torque [Nm]"] = float(s["Torque [Nm]"] + torque_drift + wear_torque)

#     rpm_drift = float(rng.normal(params["rpm_mu"], params["rpm_sigma"]))
#     s["Rotational speed [rpm]"] = float(s["Rotational speed [rpm]"] + rpm_drift)

#     # We clip all values to the training datas realistic range (roughly p2 to p98)
#     # because if the simulator drifts outside what the model has seen, it would
#     # always predict 100% risk which makes the dashboard useless.
#     s["Air temperature [K]"]     = float(np.clip(s["Air temperature [K]"],     296.0, 304.0))
#     s["Process temperature [K]"] = float(np.clip(s["Process temperature [K]"], 307.0, 313.5))
#     s["Torque [Nm]"]             = float(np.clip(s["Torque [Nm]"],              10.0,  63.0))
#     s["Rotational speed [rpm]"]  = float(np.clip(s["Rotational speed [rpm]"],  1200.0, 2100.0))
#     s["Tool wear [min]"]         = float(np.clip(s["Tool wear [min]"],            0.0,  260.0))

#     return s


# def get_or_create_machine_state(product_id: str, base_sensor: dict, step: bool = True) -> dict:
#     # We track each machines state separately so when the user switches
#     # between product IDs, each machine remembers where it left off
#     # instead of resetting to its original dataset values.
#     machines = st.session_state.machines
#     tick_now = int(st.session_state.sim_tick)

#     if product_id not in machines:
#         init_state = _attach_sim_internals(base_sensor)
#         machines[product_id] = {"state": init_state, "last_tick": tick_now}
#         return init_state

#     record = machines[product_id]
#     state = record["state"]
#     last_tick = int(record.get("last_tick", tick_now))

#     if not step:
#         return state

#     # We advance by however many ticks were missed since this machine
#     # was last updated. Usually this is 0 or 1 but it can be more if
#     # the user was viewing a different machine for several ticks.
#     steps = max(0, tick_now - last_tick)
#     for _ in range(steps):
#         state = _advance_with_maintenance(state)

#     record["state"] = state
#     record["last_tick"] = tick_now
#     machines[product_id] = record

#     return state


# def reset_simulation():
#     st.session_state.machines = {}
#     st.session_state.sim_tick = 0
#     st.session_state.history = []


# def step_all_machines(df_source: pd.DataFrame):
#     # We only advance the machines the user has explicitly selected via
#     # the sidebar dropdowns. Stepping the full 10 000-row dataset every
#     # tick was the primary cause of slow refreshes — now we touch at
#     # most 10 rows regardless of dataset size.
#     tick_now = int(st.session_state.sim_tick)
#     machines = st.session_state.machines

#     selected_pids = [str(p) for p in st.session_state.get("sim_selected_pids", [])]
#     if not selected_pids:
#         return

#     # Build an index once so look-ups are O(1) instead of a full scan.
#     # We keep a separate dict of rows with Product ID restored because
#     # set_index() removes "Product ID" as a column, which makes
#     # build_sensor_from_row raise a KeyError when it tries row["Product ID"].
#     df_indexed = df_source.set_index("Product ID")

#     for pid in selected_pids:
#         if pid not in df_indexed.index:
#             continue

#         if pid not in machines:
#             # .loc[pid] returns a Series if unique, DataFrame if duplicates
#             raw = df_indexed.loc[pid]
#             if isinstance(raw, pd.DataFrame):
#                 raw = raw.iloc[0]
#             row = raw.copy()
#             row["Product ID"] = pid   # restore the column removed by set_index
#             base = build_sensor_from_row(row)
#             init_state = _attach_sim_internals(base)
#             machines[pid] = {"state": init_state, "last_tick": tick_now}
#             continue

#         record = machines[pid]
#         state = record["state"]
#         last_tick = int(record.get("last_tick", tick_now))

#         steps = max(0, tick_now - last_tick)
#         for _ in range(steps):
#             state = _advance_with_maintenance(state)

#         record["state"] = state
#         record["last_tick"] = tick_now
#         machines[pid] = record


# def render_common_kpis_and_gauge(sensor: dict, top_k: int):
#     # Lazy imports here break the circular dependency chain:
#     # dashboard_utils → inference → (indirectly) dashboard_utils
#     from src.inference import predict, compute_ttf_proxy  # noqa: PLC0415
#     from src.shap_explain import get_top_shap_drivers      # noqa: PLC0415
#     # We run prediction and TTF estimation here because all three
#     # pages need to display the same KPI cards and gauge.
#     model_input = {
#         k: v for k, v in sensor.items()
#         if k != "Product ID" and not str(k).startswith("_")
#     }

#     maintenance_active = bool(sensor.get("_maintenance_active", False))
#     operational_status = "Under Maintenance" if maintenance_active else "Operational"

#     if maintenance_active:
#         risk_prob = None
#         risk_label = "N/A"

#         ttf_info = compute_ttf_proxy(model_input)
#         ttf_value = float(ttf_info.get("ttf_min", 0.0))
#         ttf_method = str(ttf_info.get("method", "unknown"))

#         shap_drivers = []
#     else:
#         result = predict(model_input)

#         risk_prob = float(result.get("risk_probability", 0.0))
#         risk_label = str(result.get("risk_label", "N/A"))

#         ttf_info = compute_ttf_proxy(model_input)
#         ttf_value = float(ttf_info.get("ttf_min", 0.0))
#         ttf_method = str(ttf_info.get("method", "unknown"))

#         shap_drivers = []
#         try:
#             shap_drivers = get_top_shap_drivers(model_input, top_k=top_k)
#         except Exception as e:
#             st.warning(f"SHAP drivers not available: {e}")

#     k1, k2, k3, k4, k5 = st.columns(5)
#     k1.metric("Product ID", sensor.get("Product ID", "N/A"))
#     k2.metric("Risk Probability", "N/A" if maintenance_active else f"{risk_prob:.2%}")
#     k3.metric("Risk Level", "N/A" if maintenance_active else risk_label)
#     k4.metric("TTF (min)", "N/A" if maintenance_active else f"{ttf_value:.1f}", help=f"Method: {ttf_method}")
#     k5.metric("Operational Status", operational_status)

#     if ttf_method == "wear_rule_fallback":
#         k4.caption("Fallback estimate")
#     else:
#         k4.caption("Regression model")

#     g1, g2, g3 = st.columns([1, 2, 1])
#     with g2:
#         if maintenance_active:
#             st.markdown(
#                 """
#                 <div style="
#                     height:320px;
#                     display:flex;
#                     align-items:center;
#                     justify-content:center;
#                     border:1px solid rgba(255,255,255,0.15);
#                     border-radius:12px;
#                     background-color:rgba(255,255,255,0.02);
#                     font-size:28px;
#                     font-weight:600;
#                     color:white;
#                 ">
#                     Failure Risk Gauge: N/A
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
#         else:
#             threshold = float(result.get("threshold_used", 0.18))
#             st.plotly_chart(make_risk_gauge(risk_prob, threshold=threshold), use_container_width=True)

#     return (0.0 if risk_prob is None else risk_prob), shap_drivers


# def _df_fingerprint(df: pd.DataFrame) -> str:
#     # We hash the dataframe so Streamlits cache knows when the Cox
#     # model needs to be refit. Without this, it would either refit
#     # every refresh (slow) or never update after page 2 edits.
#     h = pd.util.hash_pandas_object(df, index=False).values
#     return f"{int(h.sum())}_{len(df)}_{len(df.columns)}"


# @st.cache_resource(show_spinner=False)
# def _fit_cox_cached(df_cox: pd.DataFrame, _fp: str):
#     # We import lifelines lazily here so pages 1 and 2 dont crash
#     # if lifelines is not installed on the users machine.
#     from src.survival_analysis import fit_cox_model
#     return fit_cox_model(df_cox)


# # src/dashboard_utils.py
# from __future__ import annotations
# from pathlib import Path
# import csv as _csv
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# import streamlit as st

# MAINTENANCE_TRIGGER_PROB  = 0.90
# MAINTENANCE_TICKS_BY_TYPE = {"H": 20, "M": 15, "L": 12}

# ROOT = Path(__file__).resolve().parents[1]

# DATA_PATH     = ROOT / "data" / "cleaned" / "ai4i2020_cleaned.csv"
# FALLBACK_PATHS= [ROOT / "ai4i2020_cleaned.csv", Path("/mnt/data/ai4i2020_cleaned.csv")]
# FAILURE_COLS  = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]

# SNAPSHOT_PATH = ROOT / "data" / "latest_snapshot.csv"

# # Schema written on every Pause — matches ai4i2020.csv exactly
# SNAPSHOT_COLUMNS = [
#     "UDI", "Product ID", "Type",
#     "Air temperature [K]", "Process temperature [K]",
#     "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
#     "Machine failure",
# ]


# # ── Matplotlib dark helpers ───────────────────────────────────────────────────

# def apply_dark_mpl(ax, fig=None):
#     if fig is not None:
#         fig.patch.set_alpha(0.0)
#         fig.patch.set_facecolor((0, 0, 0, 0))
#     ax.set_facecolor((0, 0, 0, 0))
#     ax.title.set_color("white")
#     ax.xaxis.label.set_color("white")
#     ax.yaxis.label.set_color("white")
#     ax.tick_params(axis="x", colors="white")
#     ax.tick_params(axis="y", colors="white")
#     for spine in ax.spines.values():
#         spine.set_color("white")
#         spine.set_alpha(0.35)
#     ax.grid(True, alpha=0.18)
#     leg = ax.get_legend()
#     if leg is not None:
#         frame = leg.get_frame()
#         frame.set_alpha(0.25)
#         frame.set_edgecolor("white")
#         for t in leg.get_texts():
#             t.set_color("white")


# def finalize_fig(fig):
#     fig.set_size_inches(6.0, 4.0)
#     fig.tight_layout(pad=1.0)
#     return fig


# # ── Dataset loading ───────────────────────────────────────────────────────────

# @st.cache_data(show_spinner=False)
# def _load_original_dataset() -> pd.DataFrame:
#     """
#     Always loads the original cleaned CSV, never the snapshot.
#     Used as the authoritative source for PIDs not yet in the snapshot.
#     """
#     path = DATA_PATH
#     if not path.exists():
#         for fp in FALLBACK_PATHS:
#             if fp.exists():
#                 path = fp
#                 break
#     if not path.exists():
#         raise FileNotFoundError(
#             f"Could not find cleaned dataset.\nTried:\n- {DATA_PATH}\n"
#             + "\n".join([f"- {p}" for p in FALLBACK_PATHS])
#         )
#     df = pd.read_csv(path)
#     if "Product ID" not in df.columns and "UDI" in df.columns:
#         df = df.rename(columns={"UDI": "Product ID"})
#     df["Product ID"] = df["Product ID"].astype(str)
#     return df


# @st.cache_data(show_spinner=False)
# def load_cleaned_dataset() -> pd.DataFrame:
#     """
#     Load the working dataset for the product ID dropdown and df_source.

#     When a snapshot exists:
#       - Snapshot rows (previously simulated machines) are loaded first
#         so their latest sensor states are used as starting points.
#       - Any PID that is in the original dataset but NOT in the snapshot
#         is appended from the original, so the full product list is always
#         available in the multiselect.

#     When no snapshot exists (fresh run or after Reset):
#       - The original cleaned CSV is used.

#     step_all_machines() always uses the original dataset to initialise
#     brand-new PIDs, so selecting a machine that was never simulated before
#     still picks up the correct baseline values.
#     """
#     if SNAPSHOT_PATH.exists():
#         try:
#             df_snap = pd.read_csv(SNAPSHOT_PATH)
#             if "Product ID" not in df_snap.columns and "UDI" in df_snap.columns:
#                 df_snap = df_snap.rename(columns={"UDI": "Product ID"})
#             df_snap["Product ID"] = df_snap["Product ID"].astype(str)

#             # Fill in any PIDs missing from the snapshot from the original
#             df_orig  = _load_original_dataset()
#             snap_pids = set(df_snap["Product ID"].tolist())
#             df_extra  = df_orig[~df_orig["Product ID"].isin(snap_pids)]
#             return pd.concat([df_snap, df_extra], ignore_index=True)
#         except Exception:
#             pass  # Fall through if snapshot is corrupt

#     return _load_original_dataset()


# # ── Sensor helpers ────────────────────────────────────────────────────────────

# def build_sensor_from_row(row: pd.Series) -> dict:
#     return {
#         "Product ID":              str(row["Product ID"]),
#         "Type":                    str(row["Type"]),
#         "Air temperature [K]":     float(row["Air temperature [K]"]),
#         "Process temperature [K]": float(row["Process temperature [K]"]),
#         "Rotational speed [rpm]":  float(row["Rotational speed [rpm]"]),
#         "Torque [Nm]":             float(row["Torque [Nm]"]),
#         "Tool wear [min]":         float(row["Tool wear [min]"]),
#     }


# def sensor_table(sensor: dict) -> pd.DataFrame:
#     clean = {k: v for k, v in sensor.items() if not k.startswith("_")}
#     return pd.DataFrame([{"Feature": k, "Value": v} for k, v in clean.items()])


# # ── Plotly charts ─────────────────────────────────────────────────────────────

# def make_risk_gauge(prob: float, threshold: float = 0.18):
#     value = float(np.clip(prob * 100.0, 0.0, 100.0))
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=value,
#         number={"suffix": "%", "valueformat": ".1f"},
#         title={"text": "Failure Risk Gauge (%)"},
#         gauge={
#             "axis": {"range": [0, 100]},
#             "steps": [
#                 {"range": [0,  35], "color": "#2ecc71"},
#                 {"range": [35, 70], "color": "#f1c40f"},
#                 {"range": [70, 100],"color": "#e74c3c"},
#             ],
#             "threshold": {
#                 "line": {"color": "white", "width": 3},
#                 "thickness": 0.85,
#                 "value": threshold * 100,
#             },
#         },
#     ))
#     fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
#     return fig


# def make_trend_chart(hist_df: pd.DataFrame, title: str):
#     if hist_df.empty:
#         return None
#     hist_df = hist_df.sort_values("ts")
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=hist_df["ts"], y=hist_df["risk_probability"],
#         mode="lines+markers", name="Risk Probability",
#     ))
#     fig.update_layout(
#         title=title, height=300, margin=dict(l=10, r=10, t=40, b=10),
#         xaxis_title="Time", yaxis_title="Risk Probability",
#         yaxis=dict(range=[0, 1]),
#     )
#     return fig


# # ── Session state ─────────────────────────────────────────────────────────────

# def _init_sim_state():
#     defaults = {
#         "machines":          {},
#         "sim_tick":          0,
#         "sim_running":       False,
#         "history":           [],
#         "df_edit":           None,
#         "df_edit_draft":     None,
#         "page1_pid":         None,
#         "page2_pid":         None,
#         "page3_pid":         None,
#         "drift_by_id":       {},
#         "sim_num_machines":  1,
#         "sim_selected_pids": [],
#     }
#     for key, val in defaults.items():
#         if key not in st.session_state:
#             st.session_state[key] = val


# # ── Fleet medians (for post-maintenance reset) ────────────────────────────────

# @st.cache_data(show_spinner=False)
# def _get_fleet_medians() -> dict:
#     """
#     Compute fleet-wide medians for Torque and Rotational speed once and cache.
#     Uses the original cleaned file (not snapshot) to keep medians stable.
#     """
#     path = DATA_PATH
#     if not path.exists():
#         for fp in FALLBACK_PATHS:
#             if fp.exists():
#                 path = fp
#                 break
#     df = pd.read_csv(path)
#     return {
#         "Torque [Nm]":            float(df["Torque [Nm]"].median()),
#         "Rotational speed [rpm]": float(df["Rotational speed [rpm]"].median()),
#     }


# # ── Simulator internals ───────────────────────────────────────────────────────

# def _attach_sim_internals(sensor: dict) -> dict:
#     s = dict(sensor)
#     s.setdefault("_t", 0)
#     s.setdefault("_air_target", float(s["Air temperature [K]"]))
#     s.setdefault("_rpm_base",   float(s["Rotational speed [rpm]"]))
#     s.setdefault("_workload",   0.5)
#     s.setdefault("_last_shock", 0.0)
#     s.setdefault("_base_Air temperature [K]",     float(s["Air temperature [K]"]))
#     s.setdefault("_base_Process temperature [K]", float(s["Process temperature [K]"]))
#     s.setdefault("_base_Rotational speed [rpm]",  float(s["Rotational speed [rpm]"]))
#     s.setdefault("_base_Torque [Nm]",             float(s["Torque [Nm]"]))
#     s.setdefault("_base_Tool wear [min]",         float(s["Tool wear [min]"]))
#     s.setdefault("_maintenance_active",     False)
#     s.setdefault("_maintenance_ticks_left", 0)
#     return s


# def _get_maintenance_duration(sensor: dict) -> int:
#     mtype = str(sensor.get("Type", "M")).upper()
#     return int(MAINTENANCE_TICKS_BY_TYPE.get(mtype, 15))


# def _start_maintenance(sensor: dict) -> dict:
#     sensor["_maintenance_active"]     = True
#     sensor["_maintenance_ticks_left"] = _get_maintenance_duration(sensor)
#     return sensor


# def _finish_maintenance(sensor: dict) -> dict:
#     medians = _get_fleet_medians()
#     sensor["Air temperature [K]"]     = float(sensor.get("_base_Air temperature [K]",     sensor["Air temperature [K]"]))
#     sensor["Process temperature [K]"] = float(sensor.get("_base_Process temperature [K]", sensor["Process temperature [K]"]))
#     sensor["Rotational speed [rpm]"]  = medians["Rotational speed [rpm]"]
#     sensor["Torque [Nm]"]             = medians["Torque [Nm]"]
#     sensor["Tool wear [min]"]         = 0.0
#     sensor["_air_target"]             = float(sensor["Air temperature [K]"])
#     sensor["_rpm_base"]               = medians["Rotational speed [rpm]"]
#     sensor["_workload"]               = 0.5
#     sensor["_last_shock"]             = 0.0
#     sensor["_maintenance_active"]     = False
#     sensor["_maintenance_ticks_left"] = 0
#     return sensor


# def _advance_with_maintenance(sensor: dict) -> dict:
#     if bool(sensor.get("_maintenance_active", False)):
#         ticks_left = int(sensor.get("_maintenance_ticks_left", 0)) - 1
#         sensor["_maintenance_ticks_left"] = max(0, ticks_left)
#         if sensor["_maintenance_ticks_left"] <= 0:
#             sensor = _finish_maintenance(sensor)
#         sensor["_t"] = int(sensor.get("_t", 0)) + 1
#         return sensor

#     from src.simulator import step_sensor_state
#     sensor = step_sensor_state(sensor)

#     model_input = {
#         "Type":                    sensor.get("Type", "M"),
#         "Air temperature [K]":     float(sensor.get("Air temperature [K]",     0)),
#         "Process temperature [K]": float(sensor.get("Process temperature [K]", 0)),
#         "Rotational speed [rpm]":  float(sensor.get("Rotational speed [rpm]",  0)),
#         "Torque [Nm]":             float(sensor.get("Torque [Nm]",             0)),
#         "Tool wear [min]":         float(sensor.get("Tool wear [min]",          0)),
#     }
#     try:
#         from src.inference import predict
#         result = predict(model_input)
#         if float(result.get("risk_probability", 0.0)) >= MAINTENANCE_TRIGGER_PROB:
#             sensor = _start_maintenance(sensor)
#     except Exception:
#         pass

#     return sensor


# def _get_machine_params(product_id: str, base_sensor: dict, rng: np.random.Generator) -> dict:
#     if product_id in st.session_state.drift_by_id:
#         return st.session_state.drift_by_id[product_id]

#     mtype = str(base_sensor.get("Type", "M")).upper()

#     if mtype == "H":
#         wear_range = (0.3, 0.9)
#         torque_mu, torque_sigma = 0.04, 0.18
#         air_mu, air_sigma       = 0.01, 0.06
#         proc_mu, proc_sigma     = 0.02, 0.08
#         rpm_mu, rpm_sigma       = -0.5, 5.0
#     elif mtype == "L":
#         wear_range = (0.1, 0.5)
#         torque_mu, torque_sigma = 0.02, 0.10
#         air_mu, air_sigma       = 0.01, 0.04
#         proc_mu, proc_sigma     = 0.01, 0.05
#         rpm_mu, rpm_sigma       = -0.3, 3.0
#     else:
#         wear_range = (0.2, 0.7)
#         torque_mu, torque_sigma = 0.03, 0.14
#         air_mu, air_sigma       = 0.01, 0.05
#         proc_mu, proc_sigma     = 0.01, 0.06
#         rpm_mu, rpm_sigma       = -0.4, 4.0

#     params = {
#         "wear_per_tick":    float(rng.uniform(*wear_range)),
#         "air_mu":           float(rng.normal(air_mu, 0.005)),
#         "air_sigma":        float(rng.uniform(air_sigma * 0.7, air_sigma * 1.1)),
#         "proc_mu":          float(rng.normal(proc_mu, 0.005)),
#         "proc_sigma":       float(rng.uniform(proc_sigma * 0.7, proc_sigma * 1.2)),
#         "torque_mu":        float(rng.normal(torque_mu, 0.01)),
#         "torque_sigma":     float(rng.uniform(torque_sigma * 0.7, torque_sigma * 1.2)),
#         "rpm_mu":           float(rng.normal(rpm_mu, 0.2)),
#         "rpm_sigma":        float(rng.uniform(rpm_sigma * 0.7, rpm_sigma * 1.2)),
#         "wear_heat_gain":   float(rng.uniform(0.0002, 0.0008)),
#         "wear_torque_gain": float(rng.uniform(0.001, 0.003)),
#     }

#     st.session_state.drift_by_id[product_id] = params
#     return params


# def _apply_one_tick(sensor: dict, params: dict, rng: np.random.Generator) -> dict:
#     s = dict(sensor)

#     s["Tool wear [min]"] = float(s["Tool wear [min]"] + params["wear_per_tick"])
#     wear = float(s["Tool wear [min]"])

#     air_drift  = float(rng.normal(params["air_mu"],   params["air_sigma"]))
#     proc_extra = float(rng.normal(params["proc_mu"],  params["proc_sigma"]))

#     wear_heat = wear * params["wear_heat_gain"]
#     s["Air temperature [K]"]     = float(s["Air temperature [K]"]     + air_drift + (0.3 * wear_heat))
#     s["Process temperature [K]"] = float(s["Process temperature [K]"] + air_drift + proc_extra + wear_heat)

#     torque_drift = float(rng.normal(params["torque_mu"], params["torque_sigma"]))
#     wear_torque  = wear * params["wear_torque_gain"]
#     s["Torque [Nm]"]             = float(s["Torque [Nm]"]             + torque_drift + wear_torque)

#     rpm_drift = float(rng.normal(params["rpm_mu"], params["rpm_sigma"]))
#     s["Rotational speed [rpm]"]  = float(s["Rotational speed [rpm]"]  + rpm_drift)

#     s["Air temperature [K]"]     = float(np.clip(s["Air temperature [K]"],     296.0, 304.0))
#     s["Process temperature [K]"] = float(np.clip(s["Process temperature [K]"], 307.0, 313.5))
#     s["Torque [Nm]"]             = float(np.clip(s["Torque [Nm]"],              10.0,  63.0))
#     s["Rotational speed [rpm]"]  = float(np.clip(s["Rotational speed [rpm]"],  1200.0, 2100.0))
#     s["Tool wear [min]"]         = float(np.clip(s["Tool wear [min]"],            0.0,  260.0))

#     return s


# # ── Machine state management ──────────────────────────────────────────────────

# def get_or_create_machine_state(product_id: str, base_sensor: dict, step: bool = True) -> dict:
#     machines = st.session_state.machines
#     tick_now = int(st.session_state.sim_tick)

#     if product_id not in machines:
#         init_state = _attach_sim_internals(base_sensor)
#         machines[product_id] = {"state": init_state, "last_tick": tick_now}
#         return init_state

#     record    = machines[product_id]
#     state     = record["state"]
#     last_tick = int(record.get("last_tick", tick_now))

#     if not step:
#         return state

#     steps = max(0, tick_now - last_tick)
#     for _ in range(steps):
#         state = _advance_with_maintenance(state)

#     record["state"]     = state
#     record["last_tick"] = tick_now
#     machines[product_id] = record
#     return state


# def reset_simulation():
#     """Full reset — clears machines, tick, history and removes snapshot so
#     the next Start loads the original cleaned dataset again."""
#     st.session_state.machines          = {}
#     st.session_state.sim_tick          = 0
#     st.session_state.history           = []
#     st.session_state.drift_by_id       = {}
#     # Remove the snapshot so next load uses the clean dataset
#     if SNAPSHOT_PATH.exists():
#         try:
#             SNAPSHOT_PATH.unlink()
#         except Exception:
#             pass
#     # Clear both caches so the next load picks up the right source
#     load_cleaned_dataset.clear()
#     _load_original_dataset.clear()


# def step_all_machines(df_source: pd.DataFrame):
#     """
#     Advance only the user-selected machines by one tick.

#     For a PID that has never been simulated before:
#       - If it exists in the snapshot (df_source), use those sensor values
#         as the starting point (the machine picks up from its last state).
#       - If it is NOT in the snapshot (brand-new selection), fall back to
#         the original cleaned dataset so we get the correct baseline values.

#     This means you can freely add new machines to the multiselect at any
#     time — they always start from the right initial sensor readings.
#     """
#     tick_now      = int(st.session_state.sim_tick)
#     machines      = st.session_state.machines
#     selected_pids = [str(p) for p in st.session_state.get("sim_selected_pids", [])]

#     if not selected_pids:
#         return

#     # Primary index: df_source (snapshot rows take priority)
#     df_indexed = df_source.set_index("Product ID")

#     # Fallback index: original dataset for PIDs not yet in the snapshot
#     df_orig_indexed = _load_original_dataset().set_index("Product ID")

#     for pid in selected_pids:
#         if pid not in machines:
#             # Choose the best available source for the initial state
#             if pid in df_indexed.index:
#                 raw = df_indexed.loc[pid]
#             elif pid in df_orig_indexed.index:
#                 raw = df_orig_indexed.loc[pid]
#             else:
#                 continue   # PID exists nowhere — skip

#             if isinstance(raw, pd.DataFrame):
#                 raw = raw.iloc[0]
#             row = raw.copy()
#             row["Product ID"] = pid
#             base       = build_sensor_from_row(row)
#             init_state = _attach_sim_internals(base)
#             machines[pid] = {"state": init_state, "last_tick": tick_now}
#             continue

#         if pid not in df_indexed.index and pid not in df_orig_indexed.index:
#             continue   # Safety: unknown PID

#         record    = machines[pid]
#         state     = record["state"]
#         last_tick = int(record.get("last_tick", tick_now))

#         steps = max(0, tick_now - last_tick)
#         for _ in range(steps):
#             state = _advance_with_maintenance(state)

#         record["state"]     = state
#         record["last_tick"] = tick_now
#         machines[pid]       = record


# # ── Snapshot: save on Pause, load on next Start ───────────────────────────────

# def save_snapshot_on_pause() -> pd.DataFrame | None:
#     """
#     Called every time the operator hits Pause.

#     1. Takes the current sensor state of every machine that has been simulated.
#     2. Runs a single vectorised model call to assign Machine failure labels.
#     3. Writes data/latest_snapshot.csv in the original ai4i2020 schema.
#        (Overwrites the previous snapshot — only the latest state is kept.)
#     4. Returns the snapshot DataFrame so the app can offer a download button.

#     On the NEXT Start, load_cleaned_dataset() will detect the snapshot and
#     automatically load from it, so the simulation continues from exactly
#     where it left off.

#     If no machines have been stepped yet (e.g. Pause pressed before Start)
#     returns None without writing anything.
#     """
#     machines = st.session_state.get("machines", {})
#     if not machines:
#         return None

#     import src.inference as _inf
#     from src.features import add_engineered_features

#     machine_ids = list(machines.keys())
#     states      = [machines[m]["state"] for m in machine_ids]

#     df_feat = pd.DataFrame([{
#         "Type":                    s.get("Type", "M"),
#         "Air temperature [K]":     float(s.get("Air temperature [K]",     0)),
#         "Process temperature [K]": float(s.get("Process temperature [K]", 0)),
#         "Rotational speed [rpm]":  float(s.get("Rotational speed [rpm]",  0)),
#         "Torque [Nm]":             float(s.get("Torque [Nm]",             0)),
#         "Tool wear [min]":         float(s.get("Tool wear [min]",          0)),
#     } for s in states])

#     _inf._load_models()
#     probs     = _inf._model.predict_proba(add_engineered_features(df_feat.copy()))[:, 1]
#     threshold = float(_inf.FINAL_THRESHOLD)

#     rows = []
#     for i, mid in enumerate(machine_ids):
#         s = states[i]
#         rows.append({
#             "UDI":                     i + 1,
#             "Product ID":              str(mid),
#             "Type":                    str(s.get("Type", "M")),
#             "Air temperature [K]":     round(float(s.get("Air temperature [K]",     0)), 4),
#             "Process temperature [K]": round(float(s.get("Process temperature [K]", 0)), 4),
#             "Rotational speed [rpm]":  round(float(s.get("Rotational speed [rpm]",  0)), 2),
#             "Torque [Nm]":             round(float(s.get("Torque [Nm]",             0)), 4),
#             "Tool wear [min]":         round(float(s.get("Tool wear [min]",          0)), 2),
#             "Machine failure":         int(float(probs[i]) >= threshold),
#         })

#     SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
#     with open(SNAPSHOT_PATH, "w", newline="") as f:
#         writer = _csv.DictWriter(f, fieldnames=SNAPSHOT_COLUMNS)
#         writer.writeheader()
#         writer.writerows(rows)

#     # Clear cache so the next load_cleaned_dataset() call reads the new snapshot
#     load_cleaned_dataset.clear()

#     return pd.DataFrame(rows)


# # ── KPI / gauge rendering ─────────────────────────────────────────────────────

# def render_common_kpis_and_gauge(sensor: dict, top_k: int):
#     from src.inference import predict, compute_ttf_proxy
#     from src.shap_explain import get_top_shap_drivers

#     model_input = {k: v for k, v in sensor.items()
#                    if k != "Product ID" and not str(k).startswith("_")}

#     maintenance_active = bool(sensor.get("_maintenance_active", False))
#     operational_status = "Under Maintenance" if maintenance_active else "Operational"

#     if maintenance_active:
#         risk_prob    = None
#         risk_label   = "N/A"
#         ttf_info     = compute_ttf_proxy(model_input)
#         ttf_value    = float(ttf_info.get("ttf_min", 0.0))
#         ttf_method   = str(ttf_info.get("method", "unknown"))
#         shap_drivers = []
#     else:
#         result       = predict(model_input)
#         risk_prob    = float(result.get("risk_probability", 0.0))
#         risk_label   = str(result.get("risk_label", "N/A"))
#         ttf_info     = compute_ttf_proxy(model_input)
#         ttf_value    = float(ttf_info.get("ttf_min", 0.0))
#         ttf_method   = str(ttf_info.get("method", "unknown"))
#         shap_drivers = []
#         try:
#             shap_drivers = get_top_shap_drivers(model_input, top_k=top_k)
#         except Exception as e:
#             st.warning(f"SHAP drivers not available: {e}")

#     k1, k2, k3, k4, k5 = st.columns(5)
#     k1.metric("Product ID",         sensor.get("Product ID", "N/A"))
#     k2.metric("Risk Probability",   "N/A" if maintenance_active else f"{risk_prob:.2%}")
#     k3.metric("Risk Level",         "N/A" if maintenance_active else risk_label)
#     k4.metric("TTF (min)",          "N/A" if maintenance_active else f"{ttf_value:.1f}", help=f"Method: {ttf_method}")
#     k5.metric("Operational Status", operational_status)

#     if ttf_method == "wear_rule_fallback":
#         k4.caption("Fallback estimate")
#     else:
#         k4.caption("Regression model")

#     g1, g2, g3 = st.columns([1, 2, 1])
#     with g2:
#         if maintenance_active:
#             st.markdown(
#                 """<div style="height:320px;display:flex;align-items:center;
#                 justify-content:center;border:1px solid rgba(255,255,255,0.15);
#                 border-radius:12px;background-color:rgba(255,255,255,0.02);
#                 font-size:28px;font-weight:600;color:white;">
#                 Failure Risk Gauge: N/A</div>""",
#                 unsafe_allow_html=True,
#             )
#         else:
#             threshold = float(result.get("threshold_used", 0.18))
#             st.plotly_chart(make_risk_gauge(risk_prob, threshold=threshold), use_container_width=True)

#     return (0.0 if risk_prob is None else risk_prob), shap_drivers


# # ── Cox helpers ───────────────────────────────────────────────────────────────

# def _df_fingerprint(df: pd.DataFrame) -> str:
#     h = pd.util.hash_pandas_object(df, index=False).values
#     return f"{int(h.sum())}_{len(df)}_{len(df.columns)}"


# @st.cache_resource(show_spinner=False)
# def _fit_cox_cached(df_cox: pd.DataFrame, _fp: str):
#     from src.survival_analysis import fit_cox_model
#     return fit_cox_model(df_cox)


# src/dashboard_utils.py
from __future__ import annotations
from pathlib import Path
import csv as _csv
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MAINTENANCE_TRIGGER_PROB  = 0.90
MAINTENANCE_TICKS_BY_TYPE = {"H": 20, "M": 15, "L": 12}

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH     = ROOT / "data" / "cleaned" / "ai4i2020_cleaned.csv"
FALLBACK_PATHS= [ROOT / "ai4i2020_cleaned.csv", Path("/mnt/data/ai4i2020_cleaned.csv")]
FAILURE_COLS  = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]

SNAPSHOT_PATH = ROOT / "data" / "latest_snapshot.csv"

# Schema written on every Pause — matches ai4i2020.csv exactly
SNAPSHOT_COLUMNS = [
    "UDI", "Product ID", "Type",
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    "Machine failure",
]


# ── Matplotlib dark helpers ───────────────────────────────────────────────────

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
    fig.set_size_inches(6.0, 4.0)
    fig.tight_layout(pad=1.0)
    return fig


# ── Dataset loading ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_original_dataset() -> pd.DataFrame:
    """
    Always loads the original cleaned CSV, never the snapshot.
    Used as the authoritative source for PIDs not yet in the snapshot.
    """
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
    if "Product ID" not in df.columns and "UDI" in df.columns:
        df = df.rename(columns={"UDI": "Product ID"})
    df["Product ID"] = df["Product ID"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_cleaned_dataset() -> pd.DataFrame:
    """
    Load the working dataset for the product ID dropdown and df_source.

    When a snapshot exists:
      - Snapshot rows (previously simulated machines) are loaded first
        so their latest sensor states are used as starting points.
      - Any PID that is in the original dataset but NOT in the snapshot
        is appended from the original, so the full product list is always
        available in the multiselect.

    When no snapshot exists (fresh run or after Reset):
      - The original cleaned CSV is used.

    step_all_machines() always uses the original dataset to initialise
    brand-new PIDs, so selecting a machine that was never simulated before
    still picks up the correct baseline values.
    """
    if SNAPSHOT_PATH.exists():
        try:
            df_snap = pd.read_csv(SNAPSHOT_PATH)
            if "Product ID" not in df_snap.columns and "UDI" in df_snap.columns:
                df_snap = df_snap.rename(columns={"UDI": "Product ID"})
            df_snap["Product ID"] = df_snap["Product ID"].astype(str)

            # Fill in any PIDs missing from the snapshot from the original
            df_orig  = _load_original_dataset()
            snap_pids = set(df_snap["Product ID"].tolist())
            df_extra  = df_orig[~df_orig["Product ID"].isin(snap_pids)]
            return pd.concat([df_snap, df_extra], ignore_index=True)
        except Exception:
            pass  # Fall through if snapshot is corrupt

    return _load_original_dataset()


# ── Sensor helpers ────────────────────────────────────────────────────────────

def build_sensor_from_row(row: pd.Series) -> dict:
    return {
        "Product ID":              str(row["Product ID"]),
        "Type":                    str(row["Type"]),
        "Air temperature [K]":     float(row["Air temperature [K]"]),
        "Process temperature [K]": float(row["Process temperature [K]"]),
        "Rotational speed [rpm]":  float(row["Rotational speed [rpm]"]),
        "Torque [Nm]":             float(row["Torque [Nm]"]),
        "Tool wear [min]":         float(row["Tool wear [min]"]),
    }


def sensor_table(sensor: dict) -> pd.DataFrame:
    clean = {k: v for k, v in sensor.items() if not k.startswith("_")}
    return pd.DataFrame([{"Feature": k, "Value": v} for k, v in clean.items()])


# ── Plotly charts ─────────────────────────────────────────────────────────────

def make_risk_gauge(prob: float, threshold: float = 0.18):
    value = float(np.clip(prob * 100.0, 0.0, 100.0))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "Failure Risk Gauge (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0,  35], "color": "#2ecc71"},
                {"range": [35, 70], "color": "#f1c40f"},
                {"range": [70, 100],"color": "#e74c3c"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.85,
                "value": threshold * 100,
            },
        },
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def make_trend_chart(hist_df: pd.DataFrame, title: str):
    if hist_df.empty:
        return None
    hist_df = hist_df.sort_values("ts")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df["ts"], y=hist_df["risk_probability"],
        mode="lines+markers", name="Risk Probability",
    ))
    fig.update_layout(
        title=title, height=300, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Time", yaxis_title="Risk Probability",
        yaxis=dict(range=[0, 1]),
    )
    return fig


# ── Session state ─────────────────────────────────────────────────────────────

def _init_sim_state():
    defaults = {
        "machines":          {},
        "sim_tick":          0,
        "sim_running":       False,
        "history":           [],
        "df_edit":           None,
        "df_edit_draft":     None,
        "page1_pid":         None,
        "page2_pid":         None,
        "page3_pid":         None,
        "drift_by_id":       {},
        "sim_num_machines":  1,
        "sim_selected_pids": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Fleet medians (for post-maintenance reset) ────────────────────────────────

@st.cache_data(show_spinner=False)
def _get_fleet_medians() -> dict:
    """
    Compute fleet-wide medians for Torque and Rotational speed once and cache.
    Uses the original cleaned file (not snapshot) to keep medians stable.
    """
    path = DATA_PATH
    if not path.exists():
        for fp in FALLBACK_PATHS:
            if fp.exists():
                path = fp
                break
    df = pd.read_csv(path)
    return {
        "Torque [Nm]":            float(df["Torque [Nm]"].median()),
        "Rotational speed [rpm]": float(df["Rotational speed [rpm]"].median()),
    }


# ── Simulator internals ───────────────────────────────────────────────────────

def _attach_sim_internals(sensor: dict) -> dict:
    s = dict(sensor)
    s.setdefault("_t", 0)
    s.setdefault("_air_target", float(s["Air temperature [K]"]))
    s.setdefault("_rpm_base",   float(s["Rotational speed [rpm]"]))
    s.setdefault("_workload",   0.5)
    s.setdefault("_last_shock", 0.0)
    s.setdefault("_base_Air temperature [K]",     float(s["Air temperature [K]"]))
    s.setdefault("_base_Process temperature [K]", float(s["Process temperature [K]"]))
    s.setdefault("_base_Rotational speed [rpm]",  float(s["Rotational speed [rpm]"]))
    s.setdefault("_base_Torque [Nm]",             float(s["Torque [Nm]"]))
    s.setdefault("_base_Tool wear [min]",         float(s["Tool wear [min]"]))
    s.setdefault("_maintenance_active",     False)
    s.setdefault("_maintenance_ticks_left", 0)
    return s


def _get_maintenance_duration(sensor: dict) -> int:
    mtype = str(sensor.get("Type", "M")).upper()
    return int(MAINTENANCE_TICKS_BY_TYPE.get(mtype, 15))


def _start_maintenance(sensor: dict) -> dict:
    sensor["_maintenance_active"]     = True
    sensor["_maintenance_ticks_left"] = _get_maintenance_duration(sensor)
    return sensor


def _finish_maintenance(sensor: dict) -> dict:
    medians = _get_fleet_medians()
    sensor["Air temperature [K]"]     = float(sensor.get("_base_Air temperature [K]",     sensor["Air temperature [K]"]))
    sensor["Process temperature [K]"] = float(sensor.get("_base_Process temperature [K]", sensor["Process temperature [K]"]))
    sensor["Rotational speed [rpm]"]  = medians["Rotational speed [rpm]"]
    sensor["Torque [Nm]"]             = medians["Torque [Nm]"]
    sensor["Tool wear [min]"]         = 0.0
    sensor["_air_target"]             = float(sensor["Air temperature [K]"])
    sensor["_rpm_base"]               = medians["Rotational speed [rpm]"]
    sensor["_workload"]               = 0.5
    sensor["_last_shock"]             = 0.0
    sensor["_maintenance_active"]     = False
    sensor["_maintenance_ticks_left"] = 0
    return sensor


def _advance_with_maintenance(sensor: dict) -> dict:
    if bool(sensor.get("_maintenance_active", False)):
        ticks_left = int(sensor.get("_maintenance_ticks_left", 0)) - 1
        sensor["_maintenance_ticks_left"] = max(0, ticks_left)
        if sensor["_maintenance_ticks_left"] <= 0:
            sensor = _finish_maintenance(sensor)
        sensor["_t"] = int(sensor.get("_t", 0)) + 1
        return sensor

    from src.simulator import step_sensor_state
    sensor = step_sensor_state(sensor)

    model_input = {
        "Type":                    sensor.get("Type", "M"),
        "Air temperature [K]":     float(sensor.get("Air temperature [K]",     0)),
        "Process temperature [K]": float(sensor.get("Process temperature [K]", 0)),
        "Rotational speed [rpm]":  float(sensor.get("Rotational speed [rpm]",  0)),
        "Torque [Nm]":             float(sensor.get("Torque [Nm]",             0)),
        "Tool wear [min]":         float(sensor.get("Tool wear [min]",          0)),
    }
    try:
        from src.inference import predict
        result = predict(model_input)
        if float(result.get("risk_probability", 0.0)) >= MAINTENANCE_TRIGGER_PROB:
            sensor = _start_maintenance(sensor)
    except Exception:
        pass

    return sensor


def _get_machine_params(product_id: str, base_sensor: dict, rng: np.random.Generator) -> dict:
    if product_id in st.session_state.drift_by_id:
        return st.session_state.drift_by_id[product_id]

    mtype = str(base_sensor.get("Type", "M")).upper()

    if mtype == "H":
        wear_range = (0.3, 0.9)
        torque_mu, torque_sigma = 0.04, 0.18
        air_mu, air_sigma       = 0.01, 0.06
        proc_mu, proc_sigma     = 0.02, 0.08
        rpm_mu, rpm_sigma       = -0.5, 5.0
    elif mtype == "L":
        wear_range = (0.1, 0.5)
        torque_mu, torque_sigma = 0.02, 0.10
        air_mu, air_sigma       = 0.01, 0.04
        proc_mu, proc_sigma     = 0.01, 0.05
        rpm_mu, rpm_sigma       = -0.3, 3.0
    else:
        wear_range = (0.2, 0.7)
        torque_mu, torque_sigma = 0.03, 0.14
        air_mu, air_sigma       = 0.01, 0.05
        proc_mu, proc_sigma     = 0.01, 0.06
        rpm_mu, rpm_sigma       = -0.4, 4.0

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

    air_drift  = float(rng.normal(params["air_mu"],   params["air_sigma"]))
    proc_extra = float(rng.normal(params["proc_mu"],  params["proc_sigma"]))

    wear_heat = wear * params["wear_heat_gain"]
    s["Air temperature [K]"]     = float(s["Air temperature [K]"]     + air_drift + (0.3 * wear_heat))
    s["Process temperature [K]"] = float(s["Process temperature [K]"] + air_drift + proc_extra + wear_heat)

    torque_drift = float(rng.normal(params["torque_mu"], params["torque_sigma"]))
    wear_torque  = wear * params["wear_torque_gain"]
    s["Torque [Nm]"]             = float(s["Torque [Nm]"]             + torque_drift + wear_torque)

    rpm_drift = float(rng.normal(params["rpm_mu"], params["rpm_sigma"]))
    s["Rotational speed [rpm]"]  = float(s["Rotational speed [rpm]"]  + rpm_drift)

    s["Air temperature [K]"]     = float(np.clip(s["Air temperature [K]"],     296.0, 304.0))
    s["Process temperature [K]"] = float(np.clip(s["Process temperature [K]"], 307.0, 313.5))
    s["Torque [Nm]"]             = float(np.clip(s["Torque [Nm]"],              10.0,  63.0))
    s["Rotational speed [rpm]"]  = float(np.clip(s["Rotational speed [rpm]"],  1200.0, 2100.0))
    s["Tool wear [min]"]         = float(np.clip(s["Tool wear [min]"],            0.0,  260.0))

    return s


# ── Machine state management ──────────────────────────────────────────────────

def get_or_create_machine_state(product_id: str, base_sensor: dict, step: bool = True) -> dict:
    machines = st.session_state.machines
    tick_now = int(st.session_state.sim_tick)

    if product_id not in machines:
        init_state = _attach_sim_internals(base_sensor)
        machines[product_id] = {"state": init_state, "last_tick": tick_now}
        return init_state

    record    = machines[product_id]
    state     = record["state"]
    last_tick = int(record.get("last_tick", tick_now))

    if not step:
        return state

    steps = max(0, tick_now - last_tick)
    for _ in range(steps):
        state = _advance_with_maintenance(state)

    record["state"]     = state
    record["last_tick"] = tick_now
    machines[product_id] = record
    return state


def reset_simulation():
    """Full reset — clears machines, tick, history and removes snapshot so
    the next Start loads the original cleaned dataset again."""
    st.session_state.machines          = {}
    st.session_state.sim_tick          = 0
    st.session_state.history           = []
    st.session_state.drift_by_id       = {}
    # Remove the snapshot so next load uses the clean dataset
    if SNAPSHOT_PATH.exists():
        try:
            SNAPSHOT_PATH.unlink()
        except Exception:
            pass
    # Clear both caches so the next load picks up the right source
    load_cleaned_dataset.clear()
    _load_original_dataset.clear()


def step_all_machines(df_source: pd.DataFrame):
    """
    Advance only the user-selected machines by one tick.

    For a PID that has never been simulated before:
      - If it exists in the snapshot (df_source), use those sensor values
        as the starting point (the machine picks up from its last state).
      - If it is NOT in the snapshot (brand-new selection), fall back to
        the original cleaned dataset so we get the correct baseline values.

    This means you can freely add new machines to the multiselect at any
    time — they always start from the right initial sensor readings.
    """
    tick_now      = int(st.session_state.sim_tick)
    machines      = st.session_state.machines
    selected_pids = [str(p) for p in st.session_state.get("sim_selected_pids", [])]

    if not selected_pids:
        return

    # Primary index: df_source (snapshot rows take priority)
    df_indexed = df_source.set_index("Product ID")

    # Fallback index: original dataset for PIDs not yet in the snapshot
    df_orig_indexed = _load_original_dataset().set_index("Product ID")

    for pid in selected_pids:
        if pid not in machines:
            # Choose the best available source for the initial state
            if pid in df_indexed.index:
                raw = df_indexed.loc[pid]
            elif pid in df_orig_indexed.index:
                raw = df_orig_indexed.loc[pid]
            else:
                continue   # PID exists nowhere — skip

            if isinstance(raw, pd.DataFrame):
                raw = raw.iloc[0]
            row = raw.copy()
            row["Product ID"] = pid
            base       = build_sensor_from_row(row)
            init_state = _attach_sim_internals(base)
            machines[pid] = {"state": init_state, "last_tick": tick_now}
            continue

        if pid not in df_indexed.index and pid not in df_orig_indexed.index:
            continue   # Safety: unknown PID

        record    = machines[pid]
        state     = record["state"]
        last_tick = int(record.get("last_tick", tick_now))

        steps = max(0, tick_now - last_tick)
        for _ in range(steps):
            state = _advance_with_maintenance(state)

        record["state"]     = state
        record["last_tick"] = tick_now
        machines[pid]       = record


# ── Snapshot: save on Pause, load on next Start ───────────────────────────────

def save_snapshot_on_pause() -> pd.DataFrame | None:
    """
    Called every time the operator hits Pause.

    Writes data/latest_snapshot.csv with ALL ~10,000 rows:
      - Simulated machines: their latest live sensor states + model-predicted
        Machine failure label.
      - Non-simulated machines: their rows taken as-is from the previous
        snapshot (if one exists) or from the original cleaned dataset.

    This means the snapshot always contains the full fleet so the next
    Start never needs to fall back to the original dataset for any PID.
    """
    machines = st.session_state.get("machines", {})
    if not machines:
        return None

    import src.inference as _inf
    from src.features import add_engineered_features

    # ── 1. Build rows for simulated machines ─────────────────────────────────
    machine_ids = list(machines.keys())
    states      = [machines[m]["state"] for m in machine_ids]

    df_feat = pd.DataFrame([{
        "Type":                    s.get("Type", "M"),
        "Air temperature [K]":     float(s.get("Air temperature [K]",     0)),
        "Process temperature [K]": float(s.get("Process temperature [K]", 0)),
        "Rotational speed [rpm]":  float(s.get("Rotational speed [rpm]",  0)),
        "Torque [Nm]":             float(s.get("Torque [Nm]",             0)),
        "Tool wear [min]":         float(s.get("Tool wear [min]",          0)),
    } for s in states])

    _inf._load_models()
    probs     = _inf._model.predict_proba(add_engineered_features(df_feat.copy()))[:, 1]
    threshold = float(_inf.FINAL_THRESHOLD)

    simulated_rows = []
    for i, mid in enumerate(machine_ids):
        s = states[i]
        simulated_rows.append({
            "UDI":                     0,           # reassigned below
            "Product ID":              str(mid),
            "Type":                    str(s.get("Type", "M")),
            "Air temperature [K]":     round(float(s.get("Air temperature [K]",     0)), 4),
            "Process temperature [K]": round(float(s.get("Process temperature [K]", 0)), 4),
            "Rotational speed [rpm]":  round(float(s.get("Rotational speed [rpm]",  0)), 2),
            "Torque [Nm]":             round(float(s.get("Torque [Nm]",             0)), 4),
            "Tool wear [min]":         round(float(s.get("Tool wear [min]",          0)), 2),
            "Machine failure":         int(float(probs[i]) >= threshold),
        })

    simulated_pids = {r["Product ID"] for r in simulated_rows}

    # ── 2. Pull non-simulated rows from previous snapshot or original CSV ─────
    # Priority: previous snapshot (already has up-to-date rows from last pause)
    # Fallback:  original cleaned dataset (first-ever pause, no snapshot yet)
    non_simulated_rows = []
    source_df = None

    if SNAPSHOT_PATH.exists():
        try:
            source_df = pd.read_csv(SNAPSHOT_PATH)
            if "Product ID" not in source_df.columns and "UDI" in source_df.columns:
                source_df = source_df.rename(columns={"UDI": "Product ID"})
            source_df["Product ID"] = source_df["Product ID"].astype(str)
        except Exception:
            source_df = None

    if source_df is None:
        source_df = _load_original_dataset()

    for _, row in source_df.iterrows():
        pid = str(row["Product ID"])
        if pid in simulated_pids:
            continue   # already covered by simulated_rows
        non_simulated_rows.append({
            "UDI":                     0,
            "Product ID":              pid,
            "Type":                    str(row.get("Type", "M")),
            "Air temperature [K]":     round(float(row.get("Air temperature [K]",     0)), 4),
            "Process temperature [K]": round(float(row.get("Process temperature [K]", 0)), 4),
            "Rotational speed [rpm]":  round(float(row.get("Rotational speed [rpm]",  0)), 2),
            "Torque [Nm]":             round(float(row.get("Torque [Nm]",             0)), 4),
            "Tool wear [min]":         round(float(row.get("Tool wear [min]",          0)), 2),
            "Machine failure":         int(row.get("Machine failure", 0)),
        })

    # ── 3. Combine: simulated first, then non-simulated, reassign UDI ─────────
    all_rows = simulated_rows + non_simulated_rows
    for i, r in enumerate(all_rows):
        r["UDI"] = i + 1

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT_PATH, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=SNAPSHOT_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    # Clear cache so the next load_cleaned_dataset() call reads the new snapshot
    load_cleaned_dataset.clear()

    return pd.DataFrame(all_rows)


# ── KPI / gauge rendering ─────────────────────────────────────────────────────

def render_common_kpis_and_gauge(sensor: dict, top_k: int):
    from src.inference import predict, compute_ttf_proxy
    from src.shap_explain import get_top_shap_drivers

    model_input = {k: v for k, v in sensor.items()
                   if k != "Product ID" and not str(k).startswith("_")}

    maintenance_active = bool(sensor.get("_maintenance_active", False))
    operational_status = "Under Maintenance" if maintenance_active else "Operational"

    if maintenance_active:
        risk_prob    = None
        risk_label   = "N/A"
        ttf_info     = compute_ttf_proxy(model_input)
        ttf_value    = float(ttf_info.get("ttf_min", 0.0))
        ttf_method   = str(ttf_info.get("method", "unknown"))
        shap_drivers = []
    else:
        result       = predict(model_input)
        risk_prob    = float(result.get("risk_probability", 0.0))
        risk_label   = str(result.get("risk_label", "N/A"))
        ttf_info     = compute_ttf_proxy(model_input)
        ttf_value    = float(ttf_info.get("ttf_min", 0.0))
        ttf_method   = str(ttf_info.get("method", "unknown"))
        shap_drivers = []
        try:
            shap_drivers = get_top_shap_drivers(model_input, top_k=top_k)
        except Exception as e:
            st.warning(f"SHAP drivers not available: {e}")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Product ID",         sensor.get("Product ID", "N/A"))
    k2.metric("Risk Probability",   "N/A" if maintenance_active else f"{risk_prob:.2%}")
    k3.metric("Risk Level",         "N/A" if maintenance_active else risk_label)
    k4.metric("TTF (min)",          "N/A" if maintenance_active else f"{ttf_value:.1f}", help=f"Method: {ttf_method}")
    k5.metric("Operational Status", operational_status)

    if ttf_method == "wear_rule_fallback":
        k4.caption("Fallback estimate")
    else:
        k4.caption("Regression model")

    g1, g2, g3 = st.columns([1, 2, 1])
    with g2:
        if maintenance_active:
            st.markdown(
                """<div style="height:320px;display:flex;align-items:center;
                justify-content:center;border:1px solid rgba(255,255,255,0.15);
                border-radius:12px;background-color:rgba(255,255,255,0.02);
                font-size:28px;font-weight:600;color:white;">
                Failure Risk Gauge: N/A</div>""",
                unsafe_allow_html=True,
            )
        else:
            threshold = float(result.get("threshold_used", 0.18))
            st.plotly_chart(make_risk_gauge(risk_prob, threshold=threshold), use_container_width=True)

    return (0.0 if risk_prob is None else risk_prob), shap_drivers


# ── Cox helpers ───────────────────────────────────────────────────────────────

def _df_fingerprint(df: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(df, index=False).values
    return f"{int(h.sum())}_{len(df)}_{len(df.columns)}"


@st.cache_resource(show_spinner=False)
def _fit_cox_cached(df_cox: pd.DataFrame, _fp: str):
    from src.survival_analysis import fit_cox_model
    return fit_cox_model(df_cox)