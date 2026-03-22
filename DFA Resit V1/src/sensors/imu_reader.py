from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

try:
    from sense_hat import SenseHat
except Exception:  # pragma: no cover
    SenseHat = None


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

        self._prepare_heading()

    def _prepare_heading(self) -> None:
        """
        Prepare a heading column for offline runs.

        Priority:
        1. Use an existing heading column if present
        2. Derive heading from velocity_x, velocity_y
        3. Derive heading from position differences
        4. Fall back to zeros
        """
        if self.heading_col in self.df.columns:
            return

        if {"velocity_x", "velocity_y"}.issubset(self.df.columns):
            vx = self.df["velocity_x"].to_numpy(dtype=float)
            vy = self.df["velocity_y"].to_numpy(dtype=float)
            heading = np.arctan2(vy, vx)

            # Forward-fill invalid or zero-velocity headings
            for i in range(len(heading)):
                if i == 0:
                    continue
                if (abs(vx[i]) < 1e-9 and abs(vy[i]) < 1e-9) or np.isnan(heading[i]):
                    heading[i] = heading[i - 1]

            if len(heading) > 1 and np.isnan(heading[0]):
                heading[0] = 0.0

            self.df[self.heading_col] = heading
            return

        if {"pos_x", "pos_y"}.issubset(self.df.columns):
            px = self.df["pos_x"].to_numpy(dtype=float)
            py = self.df["pos_y"].to_numpy(dtype=float)

            dx = np.diff(px, prepend=px[0])
            dy = np.diff(py, prepend=py[0])
            heading = np.arctan2(dy, dx)

            for i in range(1, len(heading)):
                if (abs(dx[i]) < 1e-9 and abs(dy[i]) < 1e-9) or np.isnan(heading[i]):
                    heading[i] = heading[i - 1]

            if len(heading) > 0 and np.isnan(heading[0]):
                heading[0] = 0.0

            self.df[self.heading_col] = heading
            return

        self.df[self.heading_col] = 0.0

    def step_events(self) -> Iterator[dict]:
        prev_heading = None

        for _, row in self.df.iterrows():
            is_step = bool(row[self.step_col]) if self.step_col in row.index else False
            heading = float(row[self.heading_col])
            dt = float(row[self.dt_col]) if self.dt_col in row.index else 1.0
            ts = float(row[self.timestamp_col]) if self.timestamp_col in row.index else 0.0

            if prev_heading is None:
                heading_change = 0.0
            else:
                heading_change = float(
                    np.arctan2(np.sin(heading - prev_heading), np.cos(heading - prev_heading))
                )

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