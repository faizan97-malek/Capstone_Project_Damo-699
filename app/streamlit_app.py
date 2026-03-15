import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

try:
    from streamlit import fragment as st_fragment
    _HAS_FRAGMENT = True
except ImportError:
    _HAS_FRAGMENT = False

from src.inference import predict, compute_ttf_proxy
from src.shap_explain import get_top_shap_drivers
from src.ttf_trend import estimate_ttf_trend, TrendConfig

SURVIVAL_AVAILABLE = True
try:
    from src.survival_analysis import (
        SurvivalSpec,
        build_survival_frame,
        fit_kaplan_meier,
        prepare_cox_dataframe,
        fit_cox_model,
        get_cox_hazard_ratios,
        plot_km_models,
        plot_cox_coefficients,
    )
except Exception:
    SURVIVAL_AVAILABLE = False

from src.dashboard_utils import (
    FAILURE_COLS,
    SNAPSHOT_PATH,
    apply_dark_mpl,
    finalize_fig,
    load_cleaned_dataset,
    build_sensor_from_row,
    sensor_table,
    make_risk_gauge,
    make_trend_chart,
    _init_sim_state,
    get_or_create_machine_state,
    step_all_machines,
    reset_simulation,
    render_common_kpis_and_gauge,
    save_snapshot_on_pause,
    _df_fingerprint,
    _fit_cox_cached,
)

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("Predictive Maintenance Dashboard")

# ── Load dataset (snapshot if one exists, otherwise original cleaned file) ───
df_all = load_cleaned_dataset()
_init_sim_state()

# ── Track which data source is active so the banner survives reruns ──────────
if "using_snapshot" not in st.session_state:
    st.session_state.using_snapshot = SNAPSHOT_PATH.exists()

if st.session_state.df_edit is None:
    st.session_state.df_edit = df_all.copy()
    # Update the flag whenever df_edit is freshly loaded
    st.session_state.using_snapshot = SNAPSHOT_PATH.exists()

# ── Persistent data-source banner (always visible at the top of every page) ──


df_source   = st.session_state.df_edit
product_ids = sorted(df_source["Product ID"].unique().tolist())

if st.session_state.page1_pid is None and product_ids:
    st.session_state.page1_pid = product_ids[0]
if st.session_state.page2_pid is None and product_ids:
    st.session_state.page2_pid = product_ids[0]
if st.session_state.page3_pid is None and product_ids:
    st.session_state.page3_pid = product_ids[0]

if not st.session_state.sim_selected_pids and product_ids:
    st.session_state.sim_selected_pids = [product_ids[0]]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["1) Simulation Dashboard", "2) WHAT-IF Analysis", "3) Survival Analysis"],
    index=0,
)

st.sidebar.markdown("---")
top_k = st.sidebar.slider("Top SHAP drivers to show", 3, 10, 5)

if page == "1) Simulation Dashboard":
    st.sidebar.markdown("---")
    st.sidebar.header("Simulation Controls")

    st.session_state.page1_pid = st.sidebar.selectbox(
        "Product ID (detail view)",
        product_ids,
        index=product_ids.index(st.session_state.page1_pid)
               if st.session_state.page1_pid in product_ids else 0,
        key="page1_pid_selectbox",
    )

    # Filter stored PIDs to only those present in the current product list.
    # This prevents a crash when the dataset changes (e.g. a PID that existed
    # in a previous session is not in the current options list).
    _valid_defaults = [p for p in st.session_state.sim_selected_pids
                       if p in product_ids]
    if not _valid_defaults:
        _valid_defaults = [product_ids[0]]

    selected = st.sidebar.multiselect(
        "Machines to simulate",
        options=product_ids,
        default=_valid_defaults,
        key="sim_pid_multiselect",
    )
    st.session_state.sim_selected_pids = selected
    st.session_state.sim_num_machines  = len(selected)

    if not selected:
        st.sidebar.caption("⚠️ Select at least one machine to simulate.")

    c1, c2 = st.sidebar.columns(2)
    with c1:
        start_clicked = st.button("Start", use_container_width=True)
    with c2:
        pause_clicked = st.button("Pause", use_container_width=True)

    # ── Pause: save snapshot, then stop ──────────────────────────────────────
    if pause_clicked:
        st.session_state.sim_running = False
        with st.spinner("Saving snapshot…"):
            snap_df = save_snapshot_on_pause()
        st.session_state._last_snap_df  = snap_df
        st.session_state.using_snapshot = True   # next Start will load from snapshot
        st.rerun()

    # ── Start: resume from current state (snapshot already loaded at top) ────
    if start_clicked:
        st.session_state.sim_running = True
        new_df = load_cleaned_dataset()
        st.session_state.df_edit        = new_df.copy()
        st.session_state.using_snapshot = SNAPSHOT_PATH.exists()   # reflect actual source
        st.rerun()

    status = "Running" if st.session_state.sim_running else "Paused"
    st.sidebar.caption(f"Simulation status: {status}")

    # ── Sidebar data-source indicator ─────────────────────────────────────────
    # Computed per-tick against actual snapshot contents so the label is
    # always accurate — even when the file exists but the selected machine
    # was not in it (i.e. it is brand-new and sourced from the original CSV).
    st.sidebar.markdown("---")
    _snap_pids: set = set()
    if SNAPSHOT_PATH.exists():
        try:
            _snap_pids = set(
                pd.read_csv(SNAPSHOT_PATH, usecols=["Product ID"])["Product ID"]
                .astype(str).tolist()
            )
        except Exception:
            pass

    _current_pids = set(str(p) for p in st.session_state.get("sim_selected_pids", []))
    _from_snap    = _current_pids & _snap_pids
    _from_orig    = _current_pids - _snap_pids

    if _from_snap and not _from_orig:
        st.sidebar.success("🔄 **Continuing from snapshot**")
        st.sidebar.caption("`data/latest_snapshot.csv`")
    elif _from_orig and not _from_snap:
        st.sidebar.info("🆕 **Using original dataset**")
        st.sidebar.caption("`data/cleaned/ai4i2020_cleaned.csv`")
    elif _from_snap and _from_orig:
        st.sidebar.warning("🔀 **Mixed sources**")
        st.sidebar.caption(
            f"{len(_from_snap)} from snapshot · {len(_from_orig)} from original dataset"
        )
    else:
        st.sidebar.info("🆕 **Using original dataset**")
        st.sidebar.caption("`data/cleaned/ai4i2020_cleaned.csv`")

    refresh_seconds = st.sidebar.slider("Refresh interval (seconds)", 1, 10, 2)

    if st.sidebar.button("Reset simulation"):
        reset_simulation()
        st.session_state.df_edit        = None
        st.session_state.df_edit_draft  = None
        st.session_state._last_snap_df  = None
        st.session_state.using_snapshot = False
        st.success("Simulation reset. Snapshot removed. Next Start uses original dataset.")
        st.rerun()

elif page == "2) WHAT-IF Analysis":
    st.sidebar.markdown("---")
    st.sidebar.header("Table Controls")
    st.session_state.page2_pid = st.sidebar.selectbox(
        "Product ID",
        product_ids,
        index=product_ids.index(st.session_state.page2_pid)
               if st.session_state.page2_pid in product_ids else 0,
        key="page2_pid_selectbox",
    )
    refresh_seconds = 2
    st.session_state.sim_running = False

else:
    st.sidebar.markdown("---")
    st.sidebar.header("Survival Controls")
    st.session_state.page3_pid = st.sidebar.selectbox(
        "Product ID",
        product_ids,
        index=product_ids.index(st.session_state.page3_pid)
               if st.session_state.page3_pid in product_ids else 0,
        key="page3_pid_selectbox",
    )
    refresh_seconds = 2
    st.session_state.sim_running = False

# ── Active PID / KPI sensor ───────────────────────────────────────────────────
if page == "1) Simulation Dashboard":
    active_pid = st.session_state.page1_pid
elif page == "2) WHAT-IF Analysis":
    active_pid = st.session_state.page2_pid
else:
    active_pid = st.session_state.page3_pid

kpi_row         = df_source[df_source["Product ID"] == str(active_pid)].iloc[0]
kpi_base_sensor = build_sensor_from_row(kpi_row)

if page == "1) Simulation Dashboard":
    if st.session_state.sim_running:
        st.session_state.sim_tick += 1
        step_all_machines(df_source)
    selected_pids = [str(p) for p in st.session_state.get("sim_selected_pids", [])]
    if str(active_pid) in selected_pids:
        kpi_sensor = get_or_create_machine_state(str(active_pid), kpi_base_sensor, step=False)
    else:
        kpi_sensor = kpi_base_sensor
else:
    kpi_sensor = kpi_base_sensor

risk_prob, shap_drivers = render_common_kpis_and_gauge(kpi_sensor, top_k=top_k)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Simulation Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if page == "1) Simulation Dashboard":
    st.subheader("Simulation Dashboard")

    page_content_pid = st.session_state.page1_pid
    row              = df_source[df_source["Product ID"] == str(page_content_pid)].iloc[0]
    base_sensor      = build_sensor_from_row(row)

    _selected = [str(p) for p in st.session_state.get("sim_selected_pids", [])]
    if str(page_content_pid) in _selected:
        sensor = get_or_create_machine_state(str(page_content_pid), base_sensor, step=False)
    else:
        sensor = base_sensor

    st.session_state.history.append({
        "ts":               pd.Timestamp.now(),
        "product_id":       sensor.get("Product ID", "N/A"),
        "risk_probability": float(risk_prob),
        "tick":             st.session_state.sim_tick,
    })

    # ── Live sensor + SHAP ────────────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.subheader("Live Sensor Readings")
        df_sensor = sensor_table(sensor).reset_index(drop=True)
        df_sensor.index = range(1, len(df_sensor) + 1)
        st.dataframe(df_sensor, use_container_width=True)

    with right:
        st.subheader("Key Risk Drivers (SHAP Analysis)")
        if shap_drivers:
            FEATURE_NAME_MAP = {
                "num__Torque_RPM_ratio":        "Mechanical Load Ratio",
                "num__Tool wear [min]":         "Tool Wear (min)",
                "num__Air temperature [K]":     "Air Temperature (K)",
                "num__Process temperature [K]": "Process Temperature (K)",
                "num__Rotational speed [rpm]":  "Rotational Speed (RPM)",
                "num__Temp_diff":               "Temperature Difference",
                "num__Torque [Nm]":             "Torque (Nm)",
                "cat__Type_H":                  "High-Duty Machine",
                "cat__Type_M":                  "Medium-Duty Machine",
                "cat__Type_L":                  "Low-Duty Machine",
                "Torque_RPM_ratio":             "Mechanical Load Ratio",
                "Tool wear [min]":              "Tool Wear (min)",
                "Air temperature [K]":          "Air Temperature (K)",
                "Process temperature [K]":      "Process Temperature (K)",
                "Rotational speed [rpm]":       "Rotational Speed (RPM)",
                "Temp_diff":                    "Temperature Difference",
                "Torque [Nm]":                  "Torque (Nm)",
            }
            shap_df = pd.DataFrame(shap_drivers)
            shap_df["feature"] = shap_df["feature"].map(lambda x: FEATURE_NAME_MAP.get(x, x))
            shap_df = shap_df.reset_index(drop=True)
            shap_df.index = range(1, len(shap_df) + 1)
            st.dataframe(shap_df, use_container_width=True)
        else:
            st.info("No SHAP drivers to display yet.")

    st.divider()

    # ── Risk trend chart ──────────────────────────────────────────────────────
    st.subheader("Machine Risk Trend (Live)")
    hist     = pd.DataFrame(st.session_state.history)
    pid_now  = sensor.get("Product ID", "N/A")
    hist_pid = hist[hist["product_id"] == pid_now].copy()

    trend_result = estimate_ttf_trend(
        history=st.session_state.history,
        product_id=pid_now,
        current_sensor=sensor,
    )

    t1, t2, t3 = st.columns(3)
    ttc    = trend_result.get("ticks_to_critical")
    slope  = trend_result.get("slope_per_tick")
    method = trend_result.get("method", "N/A")

    t1.metric("Ticks to Critical",
              f"{ttc:.0f}" if ttc is not None else "-",
              help="Estimated simulation ticks until risk crosses 70%")
    t2.metric("Risk Slope / Tick",
              f"{slope:+.4f}" if slope is not None else "-",
              help="Rate of risk change per tick (positive = increasing)")
    t3.metric("Trend Method",
              method.replace("_", " ").title(),
              help=trend_result.get("notes", ""))

    if hist_pid.empty or len(hist_pid) < 2:
        st.info("No trend data yet for this Product ID.")
    else:
        hist_plot = hist_pid.sort_values("ts")
        y_max_raw = hist_plot["risk_probability"].max()
        y_max     = min(max(0.10, y_max_raw * 1.3), 1.0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_plot["ts"], y=hist_plot["risk_probability"],
            mode="lines+markers", name="Risk Probability",
            line=dict(width=2), marker=dict(size=5),
        ))

        threshold = float(risk_prob) if risk_prob else 0.18
        try:
            from src.inference import FINAL_THRESHOLD
            threshold = float(FINAL_THRESHOLD)
        except Exception:
            pass

        fig.add_hline(
            y=threshold, line_dash="dash",
            line_color="rgba(231, 76, 60, 0.6)",
            annotation_text=f"Threshold ({threshold:.0%})",
            annotation_position="top left",
            annotation_font_color="rgba(231, 76, 60, 0.8)",
        )
        fig.update_layout(
            title=f"Risk Probability Trend - {pid_now}",
            height=350, margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Time", yaxis_title="Risk Probability",
            yaxis=dict(range=[0, y_max], tickformat=".1%"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── High Risk Alerts ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("High Risk Alerts")

    machines         = st.session_state.machines
    maintenance_rows = []

    if machines:
        from src.features import add_engineered_features as _add_feat

        batch_rows = []
        batch_pids = []
        for pid, record in machines.items():
            state = record["state"]
            if bool(state.get("_maintenance_active", False)):
                maintenance_rows.append({
                    "Product ID":             pid,
                    "Type":                   state.get("Type", "M"),
                    "Maintenance Ticks Left": state.get("_maintenance_ticks_left", 0),
                    "Air Temp [K]":           float(state.get("Air temperature [K]", 0)),
                    "Torque [Nm]":            float(state.get("Torque [Nm]", 0)),
                    "RPM":                    float(state.get("Rotational speed [rpm]", 0)),
                    "Tool Wear [min]":        float(state.get("Tool wear [min]", 0)),
                })
                continue

            batch_pids.append(pid)
            batch_rows.append({
                "Type":                    state.get("Type", "M"),
                "Air temperature [K]":     float(state.get("Air temperature [K]",     0)),
                "Process temperature [K]": float(state.get("Process temperature [K]", 0)),
                "Rotational speed [rpm]":  float(state.get("Rotational speed [rpm]",  0)),
                "Torque [Nm]":             float(state.get("Torque [Nm]",             0)),
                "Tool wear [min]":         float(state.get("Tool wear [min]",          0)),
            })

        if batch_rows:
            try:
                import src.inference as _inf
                _inf._load_models()

                df_batch = pd.DataFrame(batch_rows)
                df_batch = _add_feat(df_batch)
                probs    = _inf._model.predict_proba(df_batch)[:, 1]

                alert_rows = []
                for i, (pid, prob) in enumerate(zip(batch_pids, probs)):
                    r = batch_rows[i]
                    if prob >= 0.90:
                        maintenance_rows.append({
                            "Product ID":         pid,
                            "Type":               r["Type"],
                            "Maintenance Status": "Under Maintenance",
                            "Air Temp [K]":       f"{r['Air temperature [K]']:.1f}",
                            "Torque [Nm]":        f"{r['Torque [Nm]']:.1f}",
                            "RPM":                f"{r['Rotational speed [rpm]']:.0f}",
                            "Tool Wear [min]":    f"{r['Tool wear [min]']:.1f}",
                        })
                        continue
                    if prob >= 0.70:
                        ttf_info = compute_ttf_proxy(r)
                        alert_rows.append({
                            "Product ID":       pid,
                            "Type":             r["Type"],
                            "Risk Probability": f"{prob:.1%}",
                            "Risk Level":       "High",
                            "TTF (min)":        f"{ttf_info.get('ttf_min', 0):.1f}",
                            "Air Temp [K]":     f"{r['Air temperature [K]']:.1f}",
                            "Torque [Nm]":      f"{r['Torque [Nm]']:.1f}",
                            "RPM":              f"{r['Rotational speed [rpm]']:.0f}",
                            "Tool Wear [min]":  f"{r['Tool wear [min]']:.1f}",
                        })

                if alert_rows:
                    df_alerts = (
                        pd.DataFrame(alert_rows)
                        .sort_values("Risk Probability", ascending=False)
                        .reset_index(drop=True)
                    )
                    df_alerts.index = range(1, len(df_alerts) + 1)
                    st.dataframe(df_alerts, use_container_width=True)
                    st.caption(f"{len(alert_rows)} machine(s) at High risk (≥ 70% failure probability)")
                else:
                    st.info("No machines at High risk level currently.")
            except Exception as e:
                st.warning(f"Could not compute fleet risk: {e}")
        else:
            st.info("No machines initialized yet.")
    else:
        st.info("Start the simulation to monitor machine risk levels.")

    # ── Machines Under Maintenance ────────────────────────────────────────────
    st.divider()
    st.subheader("Machines Under Maintenance")

    if machines:
        if maintenance_rows:
            maint_df = pd.DataFrame(maintenance_rows).reset_index(drop=True)
            maint_df.index = range(1, len(maint_df) + 1)
            st.dataframe(maint_df, use_container_width=True)
            st.caption(f"{len(maintenance_rows)} machine(s) currently under maintenance")
        else:
            st.info("No machines currently under maintenance.")
    else:
        st.info("Start the simulation to monitor maintenance status.")

    if st.session_state.sim_running:
        time.sleep(max(0.5, refresh_seconds - 0.3))
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: WHAT-IF Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "2) WHAT-IF Analysis":
    st.subheader("Data Table")

    if st.session_state.df_edit_draft is None:
        st.session_state.df_edit_draft = df_source.copy()

    df_draft     = st.session_state.df_edit_draft
    cols_to_drop = [c for c in FAILURE_COLS if c in df_draft.columns]
    df_view      = df_draft.drop(columns=cols_to_drop, errors="ignore")

    preferred_left = [c for c in ["UDI", "Product ID", "Type"] if c in df_view.columns]
    other_cols     = [c for c in df_view.columns if c not in preferred_left]
    df_view        = df_view[preferred_left + other_cols]

    edited_df_view = st.data_editor(
        df_view,
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        key="data_editor_all",
    )

    df_updated_draft = df_draft.copy()
    for col in edited_df_view.columns:
        df_updated_draft[col] = edited_df_view[col]
    st.session_state.df_edit_draft = df_updated_draft

    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("Done", use_container_width=True):
            st.session_state.df_edit = st.session_state.df_edit_draft.copy()
            st.success("Edits applied. KPIs and gauge updated.")
            st.rerun()
    with c2:
        st.caption("Press Done to apply edits and refresh KPIs/gauge.")

    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Survival Analysis (KM + Cox)
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("<h1 style='text-align: left;'>Survival Analysis</h1>",
                unsafe_allow_html=True)

    if not SURVIVAL_AVAILABLE:
        st.error(
            "Survival module not available.\n\n"
            "Make sure you created src/survival_analysis.py and installed:\n"
            "  pip install lifelines matplotlib\n"
        )
        st.stop()

    try:
        spec   = SurvivalSpec(duration_col="Tool wear [min]",
                              event_col="Machine failure", group_col="Type")
        df_surv= build_survival_frame(df_source, spec=spec)
    except Exception as e:
        st.error(f"Could not build survival dataset: {e}")
        st.stop()

    selected_pid = str(st.session_state.page3_pid)
    selected_row = df_source[df_source["Product ID"] == selected_pid].iloc[0]

    bin_edges = [0, 50, 100, 150, 200, 250, 300, float("inf")]
    bin_labels= ["0-50","50-100","100-150","150-200","200-250","250-300","300+"]

    df_surv = df_surv.copy()
    df_surv["wear_bin"] = pd.cut(df_surv["duration"], bins=bin_edges,
                                  labels=bin_labels, right=False)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### KM (Cohort)")
        km_cohort_mode = st.selectbox(
            "KM Cohort",
            ["All machines","Type only","Wear bin only","Type + Wear bin"],
            index=0, key="km_cohort_mode_page3",
        )
        chosen_type = None
        chosen_bin  = None

        if km_cohort_mode in ["Type only","Type + Wear bin"]:
            type_options = sorted(df_surv["Type"].astype(str).unique().tolist())
            chosen_type  = st.selectbox("KM Type", type_options, index=0,
                                         key="km_type_selector_page3")
        if km_cohort_mode in ["Wear bin only","Type + Wear bin"]:
            chosen_bin = st.selectbox("KM Wear bin", bin_labels, index=0,
                                       key="km_wearbin_selector_page3")

        df_km = df_surv.copy()
        if km_cohort_mode == "Type only":
            df_km = df_km[df_km["Type"].astype(str) == str(chosen_type)].copy()
        elif km_cohort_mode == "Wear bin only":
            df_km = df_km[df_km["wear_bin"] == chosen_bin].copy()
        elif km_cohort_mode == "Type + Wear bin":
            df_km = df_km[(df_km["Type"].astype(str) == str(chosen_type)) &
                          (df_km["wear_bin"] == chosen_bin)].copy()

        if len(df_km) < 30:
            st.warning(f"Small cohort (n={len(df_km)}). KM curves may be unstable.")

        try:
            km_cohort = fit_kaplan_meier(df_km["duration"], df_km["event"],
                                          label="KM (cohort)")
            fig_km, ax = plot_km_models(
                km_cohort,
                title="KM Survival (Cohort)",
                xlabel="Tool wear [min] (proxy time)",
                ylabel="Survival probability",
            )
            show_marker = st.toggle("Show selected machine wear marker",
                                     value=False, key="km_marker_toggle_page3")
            if show_marker:
                ax.axvline(float(selected_row["Tool wear [min]"]), linestyle="--")
            apply_dark_mpl(ax, fig_km)
            finalize_fig(fig_km)
            st.pyplot(fig_km, use_container_width=True)
        except Exception as e:
            st.error(f"KM plot failed: {e}")

    with col_right:
        st.markdown("### Cox (Individual)")
        st.caption(
            f"Selected machine (for Cox): Product ID {selected_pid} | "
            f"Type {selected_row['Type']} | "
            f"Tool wear {float(selected_row['Tool wear [min]']):.1f}"
        )
        covariates = ["Air temperature [K]","Process temperature [K]",
                      "Rotational speed [rpm]","Torque [Nm]","Type"]
        try:
            selected_type = str(selected_row["Type"])
            df_surv_cox   = df_surv[df_surv["Type"].astype(str) == selected_type].copy()

            if len(df_surv_cox) < 50:
                st.warning(f"Small Cox cohort for Type={selected_type} "
                           f"(n={len(df_surv_cox)}). Coefficients may be unstable.")

            df_cox = prepare_cox_dataframe(
                df_surv_full=df_surv_cox, covariates=covariates,
                group_col="Type", duration_col="duration",
                event_col="event", drop_first=True,
            )
            cph = _fit_cox_cached(df_cox, _df_fingerprint(df_cox))

            one = pd.DataFrame([{
                "Air temperature [K]":     float(selected_row["Air temperature [K]"]),
                "Process temperature [K]": float(selected_row["Process temperature [K]"]),
                "Rotational speed [rpm]":  float(selected_row["Rotational speed [rpm]"]),
                "Torque [Nm]":             float(selected_row["Torque [Nm]"]),
                "Type":                    str(selected_row["Type"]),
            }])
            one    = pd.get_dummies(one, columns=["Type"], drop_first=True)
            X_cols = [c for c in df_cox.columns if c not in ["duration","event"]]
            one    = one.reindex(columns=X_cols, fill_value=0)
            sf     = cph.predict_survival_function(one)

            import matplotlib.pyplot as plt
            fig_ind, ax2 = plt.subplots(figsize=(6.0, 4.0))
            ax2.plot(sf.index, sf.iloc[:, 0])
            ax2.set_title(f"Cox Predicted Survival\nProduct ID {selected_pid}")
            ax2.set_xlabel("Tool wear [min] (proxy time)")
            ax2.set_ylabel("Survival probability")
            apply_dark_mpl(ax2, fig_ind)
            finalize_fig(fig_ind)
            st.pyplot(fig_ind, use_container_width=True)

            with st.expander("Hazard Ratios Table"):
                st.dataframe(get_cox_hazard_ratios(cph, sort=True, ascending=False),
                             use_container_width=True)

            with st.expander("Cox Coefficients (Log Hazard)"):
                fig_coef, axc = plot_cox_coefficients(cph, title="Cox Coefficients (log hazard)")
                apply_dark_mpl(axc, fig_coef)
                finalize_fig(fig_coef)
                st.pyplot(fig_coef, use_container_width=True)

        except Exception as e:
            st.error(f"Cox section failed: {e}")