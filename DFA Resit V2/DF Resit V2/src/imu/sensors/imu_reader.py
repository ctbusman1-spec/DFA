from __future__ import annotations

import pandas as pd

try:
    from sense_hat import SenseHat
except Exception:
    SenseHat = None


def wrap_angle(angle: float) -> float:
    import math
    return (angle + math.pi) % (2 * math.pi) - math.pi


class OfflineIMUReader:
    def __init__(
        self,
        csv_file: str,
        heading_col: str,
        step_col: str,
        dt_col: str,
        timestamp_col: str,
        fixed_step_length_m: float,
    ):
        self.df = pd.read_csv(csv_file)
        self.heading_col = heading_col
        self.step_col = step_col
        self.dt_col = dt_col
        self.timestamp_col = timestamp_col
        self.fixed_step_length_m = fixed_step_length_m

        required = [self.heading_col, self.step_col, self.dt_col, self.timestamp_col]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

    def step_events(self):
        prev_step_heading = None

        for _, row in self.df.iterrows():
            step_flag = int(row[self.step_col])
            heading = float(row[self.heading_col])
            dt = float(row[self.dt_col])
            ts = float(row[self.timestamp_col])

            if step_flag == 1:
                heading_change = 0.0 if prev_step_heading is None else wrap_angle(heading - prev_step_heading)
                prev_step_heading = heading
                yield {
                    "timestamp": ts,
                    "dt": max(dt, 1e-3),
                    "heading_change": heading_change,
                    "step_length_m": self.fixed_step_length_m,
                    "raw_heading": heading,
                }


class LiveIMUReader:
    def __init__(self, sample_rate_hz: int = 50):
        if SenseHat is None:
            raise RuntimeError("sense_hat is not available on this machine.")
        self.sense = SenseHat()
        self.sample_rate_hz = sample_rate_hz

    def get_acceleration_norm(self) -> float:
        acc = self.sense.get_accelerometer_raw()
        return float((acc["x"] ** 2 + acc["y"] ** 2 + acc["z"] ** 2) ** 0.5)

    def get_yaw(self) -> float:
        ori = self.sense.get_orientation_radians()
        return float(ori["yaw"])
