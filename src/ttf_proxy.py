from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


@dataclass
class TTFProxyConfig:
    """
    A simple, defendable proxy for 'time-to-failure'.

    Important:
    - AI4I 2020 does NOT provide true time-to-failure.
    - This proxy uses session trend + risk + tool wear to estimate a "time-to-risk-event".
    """
    low_threshold: float = 0.40     # below this = not urgent
    high_threshold: float = 0.70    # above this = urgent
    min_points_for_slope: int = 6   # need enough history to estimate trend
    horizon_minutes: float = 240.0  # cap estimate to avoid crazy numbers


def _minutes_since_start(ts_series: pd.Series) -> np.ndarray:
    base = ts_series.iloc[0]
    delta = (ts_series - base).dt.total_seconds() / 60.0
    return delta.to_numpy(dtype=float)


def estimate_ttf_proxy(
    history: List[Dict[str, Any]],
    current_sensor: Dict[str, Any],
    config: Optional[TTFProxyConfig] = None
) -> Dict[str, Any]:
    """
    Returns:
      {
        "ttf_proxy_min": float | None,
        "method": str,
        "notes": str
      }
    """
    config = config or TTFProxyConfig()

    if not history or len(history) < 2:
        return {
            "ttf_proxy_min": None,
            "method": "insufficient_history",
            "notes": "Need more live points to estimate a trend."
        }

    df = pd.DataFrame(history).copy()
    if "ts" not in df.columns or "risk_probability" not in df.columns:
        return {
            "ttf_proxy_min": None,
            "method": "missing_columns",
            "notes": "History missing ts/risk_probability."
        }

    df = df.sort_values("ts").tail(max(config.min_points_for_slope, 10))
    df["ts"] = pd.to_datetime(df["ts"])
    t = _minutes_since_start(df["ts"])
    y = df["risk_probability"].astype(float).to_numpy()

    # If we're already high risk, TTF proxy is basically now
    current_risk = float(y[-1])
    if current_risk >= config.high_threshold:
        return {
            "ttf_proxy_min": 0.0,
            "method": "already_high_risk",
            "notes": "Risk already above high threshold."
        }

    # If we're very low risk and flat, say it's far away (cap by horizon)
    if current_risk < config.low_threshold and len(df) < config.min_points_for_slope:
        return {
            "ttf_proxy_min": config.horizon_minutes,
            "method": "low_risk_default",
            "notes": "Low risk with limited history; returning capped horizon."
        }

    # Estimate slope using simple linear fit: risk = a*t + b
    # This is intentionally simple and explainable.
    if len(df) >= config.min_points_for_slope:
        a, b = np.polyfit(t, y, 1)  # slope a is risk change per minute
    else:
        a, b = 0.0, y[-1]

    # If risk is not increasing, we can't estimate crossing time reliably
    if a <= 0:
        return {
            "ttf_proxy_min": config.horizon_minutes,
            "method": "non_increasing_risk",
            "notes": "Risk is flat/decreasing; returning capped horizon."
        }

    # Predict minutes until crossing high threshold
    minutes_to_cross = (config.high_threshold - current_risk) / a
    minutes_to_cross = float(np.clip(minutes_to_cross, 0.0, config.horizon_minutes))

    # Add a small wear-based adjustment:
    # Higher tool wear -> lower TTF proxy (more urgent)
    tool_wear = current_sensor.get("Tool wear [min]", None)
    if tool_wear is not None:
        try:
            tw = float(tool_wear)
            # normalize roughly (dataset tool wear often 0-250 range)
            wear_factor = np.clip(tw / 250.0, 0.0, 1.0)
            minutes_to_cross *= (1.0 - 0.25 * wear_factor)  # up to 25% reduction
        except Exception:
            pass

    return {
        "ttf_proxy_min": round(minutes_to_cross, 1),
        "method": "trend_to_threshold",
        "notes": "Proxy based on session risk slope + wear adjustment (not true TTF)."
    }