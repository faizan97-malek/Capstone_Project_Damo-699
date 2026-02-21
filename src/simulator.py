import numpy as np
import random


def generate_sensor_state(seed: int | None = None) -> dict:
    """
    Generate a single simulated machine sensor snapshot.

    Returns
    -------
    dict
        Dictionary containing raw sensor values expected by the model.
    """

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Machine type distribution (roughly similar to dataset)
    machine_type = random.choices(
        population=["L", "M", "H"],
        weights=[0.6, 0.3, 0.1],
        k=1,
    )[0]

    # Generate realistic sensor values
    air_temp = np.random.normal(loc=300, scale=2)        # Kelvin
    process_temp = air_temp + np.random.normal(loc=10, scale=1)
    rotational_speed = np.random.normal(loc=1500, scale=200)  # rpm
    torque = np.random.normal(loc=40, scale=10)          # Nm
    tool_wear = max(0, np.random.normal(loc=50, scale=30))  # minutes

    sensor_state = {
        "Type": machine_type,
        "Air temperature [K]": float(air_temp),
        "Process temperature [K]": float(process_temp),
        "Rotational speed [rpm]": float(rotational_speed),
        "Torque [Nm]": float(torque),
        "Tool wear [min]": float(tool_wear),
    }

    return sensor_state


# Quick manual test
if __name__ == "__main__":
    print(generate_sensor_state())