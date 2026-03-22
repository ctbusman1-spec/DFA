from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

try:
    from sense_hat import SenseHat
except Exception:  # pragma: no cover
    SenseHat = None


class OfflineIMUReader:
    def __init__(self, csv_file: str, heading_col: str, step_col: str, dt_col: str, timestamp_col: str, fixed_step_length_m: float):
        self.df = pd.read_csv(csv_file)
        self.heading_col = heading_col
        self.step_col = step_col
        self.dt_col = dt_col
        self.timestamp_col = timestamp_col
        self.fixed_step_length_m = fixed_step_length_m

    def step_events(self) -> Iterator[dict]:
        prev_heading = None
        for _, row in self.df.iterrows():
            is_step = bool(row[self.step_col])
            heading = float(row[self.heading_col])
            dt = float(row[self.dt_col]) if self.dt_col in row else 1.0
            ts = float(row[self.timestamp_col]) if self.timestamp_col in row else 0.0
            if prev_heading is None:
                heading_change = 0.0
            else:
                heading_change = float(np.arctan2(np.sin(heading - prev_heading), np.cos(heading - prev_heading)))
            prev_heading = heading
            if is_step:
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
        return float(np.sqrt(acc["x"] ** 2 + acc["y"] ** 2 + acc["z"] ** 2))

    def get_yaw(self) -> float:
        ori = self.sense.get_orientation_radians()
        return float(ori["yaw"])
