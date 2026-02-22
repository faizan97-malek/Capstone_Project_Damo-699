from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Missing survival dependencies. Install with:\n"
        "  pip install lifelines\n"
        "Optional for plotting:\n"
        "  pip install matplotlib\n"
    ) from e


FAILURE_FLAG_COLS_DEFAULT: List[str] = ["TWF", "HDF", "PWF", "OSF", "RNF"]


@dataclass(frozen=True)
class SurvivalSpec:
    """Configuration for building survival data."""
    duration_col: str = "Tool wear [min]"
    event_col: str = "Machine failure"
    group_col: str = "Type"
    failure_flag_cols: Tuple[str, ...] = tuple(FAILURE_FLAG_COLS_DEFAULT)
    min_duration: float = 0.1  # replace zeros with this for model stability


def build_survival_frame(
    df: pd.DataFrame,
    spec: SurvivalSpec = SurvivalSpec(),
) -> pd.DataFrame:
    """
    Build a survival-ready dataframe with:
      - duration: proxy time (Tool wear [min])
      - event: failure indicator (1/0)
      - group: (optional) e.g., Type

    Event logic:
      - If spec.event_col exists: use it as event (cast to int)
      - Else if any failure flags exist (TWF/HDF/PWF/OSF/RNF): event = any(flag==1)
      - Else: raise

    Returns a COPY of df with added columns:
      - 'duration'
      - 'event'
    """
    if spec.duration_col not in df.columns:
        raise ValueError(f"Missing duration column: {spec.duration_col}")

    out = df.copy()

    # Build event
    if spec.event_col in out.columns:
        out["event"] = pd.to_numeric(out[spec.event_col], errors="coerce").fillna(0).astype(int)
    else:
        present_flags = [c for c in spec.failure_flag_cols if c in out.columns]
        if not present_flags:
            raise ValueError(
                f"Missing '{spec.event_col}' and no failure flag columns found. "
                f"Expected one of: {list(spec.failure_flag_cols)}"
            )
        # Any failure flag -> event = 1
        flags_numeric = out[present_flags].apply(pd.to_numeric, errors="coerce").fillna(0)
        out["event"] = (flags_numeric.sum(axis=1) > 0).astype(int)

    # Build duration
    out["duration"] = pd.to_numeric(out[spec.duration_col], errors="coerce")

    # Clean
    keep_cols = ["duration", "event"]
    if spec.group_col in out.columns:
        keep_cols.append(spec.group_col)

    out = out.dropna(subset=["duration", "event"])
    out = out[out["duration"] >= 0].copy()

    # Avoid zero-duration rows (KM/Cox can behave badly with zeros)
    out.loc[out["duration"] == 0, "duration"] = spec.min_duration

    # Ensure int for event
    out["event"] = out["event"].astype(int)

    return out


# ---------------------------
# Kaplan–Meier
# ---------------------------

def fit_kaplan_meier(
    durations: Union[pd.Series, np.ndarray],
    events: Union[pd.Series, np.ndarray],
    label: str = "All",
) -> KaplanMeierFitter:
    """Fit a Kaplan–Meier estimator and return the fitted object."""
    kmf = KaplanMeierFitter()
    kmf.fit(durations=durations, event_observed=events, label=label)
    return kmf


def fit_kaplan_meier_by_group(
    df_surv: pd.DataFrame,
    group_col: str = "Type",
    duration_col: str = "duration",
    event_col: str = "event",
) -> Dict[str, KaplanMeierFitter]:
    """
    Fit Kaplan–Meier models for each group.
    Returns dict: {group_value: fitted_kmf}
    """
    if group_col not in df_surv.columns:
        raise ValueError(f"Missing group column: {group_col}")
    if duration_col not in df_surv.columns or event_col not in df_surv.columns:
        raise ValueError(f"df_surv must include '{duration_col}' and '{event_col}'")

    models: Dict[str, KaplanMeierFitter] = {}
    for g in sorted(df_surv[group_col].astype(str).unique()):
        sub = df_surv[df_surv[group_col].astype(str) == g]
        models[g] = fit_kaplan_meier(sub[duration_col], sub[event_col], label=f"{group_col}={g}")
    return models


# ---------------------------
# Log-rank tests
# ---------------------------

def logrank_test_multigroup(
    df_surv: pd.DataFrame,
    group_col: str = "Type",
    duration_col: str = "duration",
    event_col: str = "event",
):
    """
    Multivariate log-rank test across all groups.
    Returns lifelines statistical result object.
    """
    if group_col not in df_surv.columns:
        raise ValueError(f"Missing group column: {group_col}")

    return multivariate_logrank_test(
        df_surv[duration_col],
        df_surv[group_col],
        df_surv[event_col],
    )


def logrank_test_pairwise(
    df_surv: pd.DataFrame,
    group_col: str = "Type",
    duration_col: str = "duration",
    event_col: str = "event",
) -> pd.DataFrame:
    """
    Pairwise log-rank tests between groups.
    Returns a DataFrame of p-values (lifelines output converted to df).
    """
    if group_col not in df_surv.columns:
        raise ValueError(f"Missing group column: {group_col}")

    res = pairwise_logrank_test(
        df_surv[duration_col],
        df_surv[group_col],
        df_surv[event_col],
    )
    # res.p_value is a DataFrame-like table
    return res.p_value.copy()


# ---------------------------
# Cox Proportional Hazards
# ---------------------------

def prepare_cox_dataframe(
    df_surv_full: pd.DataFrame,
    covariates: Iterable[str],
    group_col: Optional[str] = "Type",
    duration_col: str = "duration",
    event_col: str = "event",
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    Prepare a dataframe for CoxPH fitting:
      - selects covariates + duration + event
      - one-hot encodes group_col if included in covariates (or if group_col is not None and present)
      - ensures numeric columns

    NOTE: Do NOT include the proxy time variable (Tool wear) as a covariate if it is your duration axis.
    """
    covariates = list(covariates)

    # Ensure required cols
    for c in [duration_col, event_col]:
        if c not in df_surv_full.columns:
            raise ValueError(f"Missing required survival column: {c}")

    missing = [c for c in covariates if c not in df_surv_full.columns]
    if missing:
        raise ValueError(f"Missing covariates in dataframe: {missing}")

    df_cox = df_surv_full[covariates + [duration_col, event_col]].copy()

    # One-hot encode group column if it exists and is in df_cox and is non-numeric
    if group_col and group_col in df_cox.columns:
        df_cox[group_col] = df_cox[group_col].astype(str)
        df_cox = pd.get_dummies(df_cox, columns=[group_col], drop_first=drop_first)

    # Coerce to numeric where possible (lifelines requires numeric)
    for col in df_cox.columns:
        if col in (duration_col, event_col):
            continue
        df_cox[col] = pd.to_numeric(df_cox[col], errors="coerce")

    df_cox = df_cox.dropna().copy()

    # Ensure types
    df_cox[duration_col] = pd.to_numeric(df_cox[duration_col], errors="coerce")
    df_cox[event_col] = pd.to_numeric(df_cox[event_col], errors="coerce").astype(int)
    df_cox = df_cox.dropna(subset=[duration_col, event_col])
    df_cox = df_cox[df_cox[duration_col] > 0].copy()

    return df_cox


def fit_cox_model(
    df_cox: pd.DataFrame,
    duration_col: str = "duration",
    event_col: str = "event",
    penalizer: float = 0.0,
    l1_ratio: float = 0.0,
) -> CoxPHFitter:
    """
    Fit Cox proportional hazards model and return fitted CoxPHFitter.

    penalizer/l1_ratio can help if you see convergence issues.
    Example:
      fit_cox_model(df_cox, penalizer=0.1, l1_ratio=0.0)
    """
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(df_cox, duration_col=duration_col, event_col=event_col)
    return cph


def get_cox_hazard_ratios(
    cph: CoxPHFitter,
    sort: bool = True,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Return a clean hazard ratio table from a fitted CoxPHFitter.
    Columns:
      - hazard_ratio (exp(coef))
      - p_value
      - coef (log hazard)
      - se(coef)
      - z
      - ci_lower, ci_upper (if available)
    """
    s = cph.summary.copy()

    # Normalize column names across lifelines versions
    out = pd.DataFrame(index=s.index)
    if "exp(coef)" in s.columns:
        out["hazard_ratio"] = s["exp(coef)"]
    else:
        out["hazard_ratio"] = np.exp(s["coef"])

    out["coef"] = s.get("coef", np.nan)
    out["se(coef)"] = s.get("se(coef)", np.nan)
    out["z"] = s.get("z", np.nan)
    out["p_value"] = s.get("p", s.get("p_value", np.nan))

    # confidence intervals (may vary by version)
    for lo_name, hi_name in [
        ("exp(coef) lower 95%", "exp(coef) upper 95%"),
        ("exp(coef) lower 95%", "exp(coef) upper 95%"),
    ]:
        if lo_name in s.columns and hi_name in s.columns:
            out["ci_lower"] = s[lo_name]
            out["ci_upper"] = s[hi_name]
            break

    if sort:
        out = out.sort_values("hazard_ratio", ascending=ascending)

    return out


# ---------------------------
# Optional plotting helpers (matplotlib)
# ---------------------------

def plot_km_models(
    km_models: Union[KaplanMeierFitter, Dict[str, KaplanMeierFitter]],
    title: str = "Kaplan–Meier Survival",
    xlabel: str = "Time",
    ylabel: str = "Survival probability",
    grid: bool = True,
):
    """
    Plot one KM model or a dict of KM models and return (fig, ax).
    Requires matplotlib installed.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting: pip install matplotlib") from e

    fig, ax = plt.subplots(figsize=(8, 5))

    if isinstance(km_models, KaplanMeierFitter):
        km_models.plot_survival_function(ax=ax)
    else:
        for _, kmf in km_models.items():
            kmf.plot_survival_function(ax=ax)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True)

    return fig, ax


def plot_cox_coefficients(
    cph: CoxPHFitter,
    title: str = "Cox Model Coefficients (log hazard)",
    grid: bool = True,
):
    """
    Plot Cox model coefficients and return (fig, ax).
    Requires matplotlib installed.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting: pip install matplotlib") from e

    fig, ax = plt.subplots(figsize=(8, 5))
    cph.plot(ax=ax)
    ax.set_title(title)
    if grid:
        ax.grid(True)
    return fig, ax