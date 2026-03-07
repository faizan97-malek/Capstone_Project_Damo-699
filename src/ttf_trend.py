"""
Trend-based TTF estimator for the live simulation.

This module watches how risk probability evolves over simulation ticks
and estimates how many ticks remain before risk crosses a danger threshold.

This is DIFFERENT from inference.py's compute_ttf_proxy():
  - inference.py  → static, model-based: "how many operational minutes of
                     life does this machine have right now?"
  - ttf_trend.py  → dynamic, session-based: "based on how risk is trending
                     in this simulation, how many ticks until critical?"

Both are useful — one gives a snapshot, the other gives a trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TrendConfig:
    """
    Parameters for the trend-based TTF estimator.

    high_threshold : risk probability at which the machine is considered
                     critical.  Defaults to 0.70 (matches the gauge red zone),
                     but the dashboard can override this with the model's
                     tuned threshold from threshold.json.
    low_threshold  : below this, risk is too low to estimate a meaningful
                     crossing time — return the capped horizon instead.
    min_points     : minimum number of history points needed to fit a slope.
                     Fewer points → too noisy → return a default.
    max_points     : maximum number of recent points to use for the slope.
                     Using a sliding window avoids letting stale early ticks
                     drag the trend line when the simulation has been running
                     a long time.
    horizon_ticks  : cap on the returned estimate to avoid absurdly large
                     numbers when risk is rising very slowly.
    """
    high_threshold: float = 0.70
    low_threshold:  float = 0.10
    min_points:     int   = 6
    max_points:     int   = 30
    horizon_ticks:  float = 500.0


def estimate_ttf_trend(
    history: List[Dict[str, Any]],
    product_id: str,
    current_sensor: Dict[str, Any],
    config: Optional[TrendConfig] = None,
) -> Dict[str, Any]:
    """
    Estimate how many simulation ticks remain before the risk probability
    for a specific machine crosses the high threshold.

    Parameters
    ----------
    history : list[dict]
        The full session history (st.session_state.history).
        Each entry must have: product_id, tick, risk_probability.
        The 'ts' field is optional and not used for slope calculation
        (ticks are more stable than wall-clock time since the refresh
        interval is user-configurable).
    product_id : str
        Filter history to this machine only.
    current_sensor : dict
        Current sensor snapshot — used for wear-based adjustment.
    config : TrendConfig, optional
        Override default parameters.

    Returns
    -------
    dict with:
        ticks_to_critical : float | None
            Estimated ticks until risk crosses high_threshold.
            None if estimation is not possible yet.
        slope_per_tick    : float | None
            Risk change per tick (positive = risk increasing).
        current_risk      : float | None
            Most recent risk probability.
        method            : str
            Which estimation path was used.
        notes             : str
            Human-readable explanation.
    """
    config = config or TrendConfig()

    _empty = {
        "ticks_to_critical": None,
        "slope_per_tick":    None,
        "current_risk":      None,
    }

    # ------------------------------------------------------------------
    # 1. Filter history to this machine
    # ------------------------------------------------------------------
    if not history:
        return {
            **_empty,
            "method": "no_history",
            "notes":  "No simulation history yet.",
        }

    df = pd.DataFrame(history)

    required = {"product_id", "tick", "risk_probability"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        return {
            **_empty,
            "method": "missing_columns",
            "notes":  f"History missing columns: {missing}",
        }

    df = df[df["product_id"] == product_id].copy()

    if len(df) < 2:
        return {
            **_empty,
            "method": "insufficient_history",
            "notes":  "Need at least 2 data points for this machine.",
        }

    # ------------------------------------------------------------------
    # 2. Prepare the sliding window
    # ------------------------------------------------------------------
    df = df.sort_values("tick").tail(config.max_points)
    ticks = df["tick"].astype(float).to_numpy()
    risks = df["risk_probability"].astype(float).to_numpy()

    current_risk = float(risks[-1])

    # ------------------------------------------------------------------
    # 3. Early exits
    # ------------------------------------------------------------------
    if current_risk >= config.high_threshold:
        return {
            "ticks_to_critical": 0.0,
            "slope_per_tick":    None,
            "current_risk":      current_risk,
            "method":            "already_critical",
            "notes":             "Risk already at or above the high threshold.",
        }

    if current_risk < config.low_threshold and len(df) < config.min_points:
        return {
            "ticks_to_critical": config.horizon_ticks,
            "slope_per_tick":    None,
            "current_risk":      current_risk,
            "method":            "low_risk_waiting",
            "notes":             "Risk is low with limited history; returning capped horizon.",
        }

    # ------------------------------------------------------------------
    # 4. Fit slope (risk change per tick)
    # ------------------------------------------------------------------
    if len(df) >= config.min_points:
        # Simple linear fit: risk = slope * tick + intercept
        slope, _intercept = np.polyfit(ticks, risks, 1)
    else:
        slope = 0.0

    if slope <= 0:
        return {
            "ticks_to_critical": config.horizon_ticks,
            "slope_per_tick":    round(float(slope), 6),
            "current_risk":      current_risk,
            "method":            "non_increasing",
            "notes":             "Risk is flat or decreasing; returning capped horizon.",
        }

    # ------------------------------------------------------------------
    # 5. Estimate ticks to threshold crossing
    # ------------------------------------------------------------------
    ticks_to_cross = (config.high_threshold - current_risk) / slope
    ticks_to_cross = float(np.clip(ticks_to_cross, 0.0, config.horizon_ticks))

    # ------------------------------------------------------------------
    # 6. Wear-based adjustment
    #
    #    Higher tool wear → machine is closer to end-of-life → reduce
    #    the estimate slightly. This accounts for non-linear degradation
    #    that the linear slope can't capture.
    # ------------------------------------------------------------------
    wear_limits = {"H": 240.0, "M": 250.0, "L": 250.0}
    mtype = str(current_sensor.get("Type", "M")).upper()
    wear_limit = wear_limits.get(mtype, 250.0)

    tool_wear = current_sensor.get("Tool wear [min]", None)
    if tool_wear is not None:
        try:
            wear_ratio = float(np.clip(float(tool_wear) / wear_limit, 0.0, 1.0))
            # Up to 25% reduction when wear is at the limit
            ticks_to_cross *= (1.0 - 0.25 * wear_ratio)
        except (ValueError, TypeError):
            pass

    return {
        "ticks_to_critical": round(ticks_to_cross, 1),
        "slope_per_tick":    round(float(slope), 6),
        "current_risk":      current_risk,
        "method":            "trend_extrapolation",
        "notes":             "Linear trend from recent ticks + wear adjustment.",
    }