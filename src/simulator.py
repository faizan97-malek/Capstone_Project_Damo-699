from __future__ import annotations

import random
import math
from pathlib import Path

import pandas as pd

# Cache dataset in memory so we don't re-read it every refresh
_DATA_CACHE: pd.DataFrame | None = None


def _load_dataset() -> pd.DataFrame:
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE

    root = Path(__file__).resolve().parents[1]

    cleaned_path = root / "data" / "cleaned" / "ai4i2020_cleaned.csv"
    raw_path = root / "data" / "raw" / "ai4i2020.csv"

    if cleaned_path.exists():
        df = pd.read_csv(cleaned_path)
    elif raw_path.exists():
        df = pd.read_csv(raw_path)
    else:
        raise FileNotFoundError(
            "Could not find dataset. Expected one of:\n"
            f"- {cleaned_path}\n"
            f"- {raw_path}"
        )

    # Ensure Product ID exists
    if "Product ID" not in df.columns and raw_path.exists():
        raw_df = pd.read_csv(raw_path)
        if "UDI" in df.columns and "UDI" in raw_df.columns:
            df = df.merge(raw_df[["UDI", "Product ID"]], on="UDI", how="left")
        else:
            df = raw_df

    required = [
        "Product ID",
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Simulator dataset is missing columns: {missing}")

    df = df[required].dropna().reset_index(drop=True)
    _DATA_CACHE = df
    return _DATA_CACHE


def init_sensor_state(seed: int | None = None) -> dict:
    """
    Pick one real row as a starting point, then add simulator internals.
    Store the returned dict in st.session_state and repeatedly call step_sensor_state().
    """
    if seed is not None:
        random.seed(seed)

    df = _load_dataset()
    i = random.randrange(0, len(df))
    row = df.iloc[i].to_dict()

    # JSON-friendly
    row["Product ID"] = str(row["Product ID"])
    row["Type"] = str(row["Type"])
    for k in [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]:
        row[k] = float(row[k])

    # --- simulator internals (not shown to user, but keeps behavior realistic) ---
    row["_t"] = 0
    row["_air_target"] = row["Air temperature [K]"]                  # ambient target
    row["_rpm_base"] = row["Rotational speed [rpm]"]                 # base RPM level
    row["_workload"] = 0.5                                           # 0..1
    row["_last_shock"] = 0.0                                         # one-tick shock
    return row


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def step_sensor_state(state: dict, *, dt: float = 1.0) -> dict:
    """
    Advance ONE tick. Mutates and returns state.

    Behavior:
    - Air temp: mean-reverting + small noise
    - Workload: sinusoid + drift
    - RPM: cycle + workload + noise + rare dips
    - Torque: follows workload, slight inverse with RPM + rare spikes
    - Process temp: follows air + workload + torque + noise (slower response)
    - Tool wear: monotonic increasing with variable rate
    """
    # current values
    air = float(state["Air temperature [K]"])
    proc = float(state["Process temperature [K]"])
    rpm = float(state["Rotational speed [rpm]"])
    torque = float(state["Torque [Nm]"])
    wear = float(state["Tool wear [min]"])

    t = int(state.get("_t", 0))
    workload = float(state.get("_workload", 0.5))
    air_target = float(state.get("_air_target", air))
    rpm_base = float(state.get("_rpm_base", rpm))

    # -------------------------
    # 1) Workload (0..1): cycle + slow drift + noise
    # -------------------------
    cycle = 0.10 * math.sin(t / 8.0)
    drift = random.gauss(0.0, 0.01)
    workload = _clamp(workload + cycle + drift, 0.0, 1.0)

    # rare one-tick shock event
    shock = random.gauss(0.0, 1.0) if random.random() < 0.03 else 0.0

    # -------------------------
    # 2) Air temperature: mean reversion + noise
    # -------------------------
    if random.random() < 0.02:
        air_target = _clamp(air_target + random.gauss(0.0, 0.3), 295.0, 305.0)

    air = air + 0.10 * (air_target - air) + random.gauss(0.0, 0.08)
    air = _clamp(air, 290.0, 315.0)

    # -------------------------
    # 3) RPM: base + cycle + workload + noise + dips/spikes
    # -------------------------
    if random.random() < 0.02:
        rpm_base = _clamp(rpm_base + random.gauss(0.0, 40.0), 1100.0, 2200.0)

    rpm_cycle = 120.0 * math.sin(t / 6.0)
    rpm = rpm_base + rpm_cycle + 140.0 * (workload - 0.5) + random.gauss(0.0, 25.0) - 60.0 * shock

    # rare RPM dip (e.g., brief slowdown)
    if random.random() < 0.02:
        rpm -= abs(random.gauss(80.0, 25.0))

    rpm = _clamp(rpm, 800.0, 3000.0)

    # -------------------------
    # 4) Torque: workload-driven + slight inverse with RPM + spikes
    # -------------------------
    torque_target = 40.0 + 35.0 * workload + 10.0 * (2000.0 - rpm) / 2000.0
    torque = torque + 0.25 * (torque_target - torque) + random.gauss(0.0, 1.2) + 5.0 * shock

    # rare torque spike
    if random.random() < 0.02:
        torque += abs(random.gauss(6.0, 2.0))

    torque = _clamp(torque, 5.0, 90.0)

    # -------------------------
    # 5) Process temperature: follows air + workload + torque (slower)
    # -------------------------
    proc_target = air + 5.0 + 10.0 * workload + 0.05 * torque
    proc = proc + 0.18 * (proc_target - proc) + random.gauss(0.0, 0.12) + 0.3 * shock
    proc = _clamp(proc, air + 1.0, 350.0)

    # -------------------------
    # 6) Tool wear: monotonic (variable rate)
    # -------------------------
    wear_rate = (0.02 + 0.05 * workload + 0.0008 * torque) * dt
    if random.random() < 0.02:
        wear_rate *= 2.5  # occasional faster wear burst

    wear = wear + max(0.0, wear_rate) + abs(random.gauss(0.0, 0.003))
    wear = _clamp(wear, 0.0, 300.0)

    # write back
    state["Air temperature [K]"] = float(air)
    state["Process temperature [K]"] = float(proc)
    state["Rotational speed [rpm]"] = float(rpm)
    state["Torque [Nm]"] = float(torque)
    state["Tool wear [min]"] = float(wear)

    state["_t"] = t + 1
    state["_workload"] = float(workload)
    state["_air_target"] = float(air_target)
    state["_rpm_base"] = float(rpm_base)
    state["_last_shock"] = float(shock)

    return state


if __name__ == "__main__":
    s = init_sensor_state(seed=42)
    for _ in range(10):
        s = step_sensor_state(s)
        print(
            s["Air temperature [K]"],
            s["Process temperature [K]"],
            s["Rotational speed [rpm]"],
            s["Torque [Nm]"],
            s["Tool wear [min]"],
        )