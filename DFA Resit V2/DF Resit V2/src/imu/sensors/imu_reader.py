from __future__ import annotations

import math
import time
from typing import Iterator

import pandas as pd

try:
    from sense_hat import SenseHat
except Exception:
    SenseHat = None


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def circular_mean(angles) -> float:
    angles = list(angles)
    if len(angles) == 0:
        return 0.0
    s = sum(math.sin(a) for a in angles)
    c = sum(math.cos(a) for a in angles)
    return math.atan2(s, c)


class OfflineIMUReader:
    def __init__(
        self,
        csv_file: str,
        heading_col: str,
        step_col: str,
        dt_col: str,
        timestamp_col: str,
        fixed_step_length_m: float,
        gyro_col: str = "gyro_z_rads",
        gyro_bias_estimate_seconds: float = 1.5,
    ):
        self.df = pd.read_csv(csv_file)
        self.heading_col = heading_col
        self.step_col = step_col
        self.dt_col = dt_col
        self.timestamp_col = timestamp_col
        self.fixed_step_length_m = fixed_step_length_m
        self.gyro_col = gyro_col
        self.gyro_bias_estimate_seconds = gyro_bias_estimate_seconds

        required = [
            self.heading_col,
            self.step_col,
            self.dt_col,
            self.timestamp_col,
            self.gyro_col,
        ]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        # Force numeric parsing so blanks / text become NaN
        for col in required:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Step flag should be integer-like; missing means no step
        self.df[self.step_col] = self.df[self.step_col].fillna(0)

        # Timing columns must exist and be valid
        self.df = self.df.dropna(subset=[self.dt_col, self.timestamp_col])

        # Heading: fill gaps from nearby values
        self.df[self.heading_col] = self.df[self.heading_col].ffill().bfill()

        # Gyro: missing values treated as no rotation
        self.df[self.gyro_col] = self.df[self.gyro_col].fillna(0.0)

        self.df = self.df.reset_index(drop=True)

        self.gyro_bias = self.estimate_gyro_bias(self.gyro_bias_estimate_seconds)

    def estimate_initial_heading(self, average_seconds: float = 1.5) -> float:
        if self.df.empty:
            return 0.0

        t0 = float(self.df[self.timestamp_col].iloc[0])
        mask = self.df[self.timestamp_col] <= (t0 + average_seconds)
        headings = self.df.loc[mask, self.heading_col].astype(float).tolist()

        headings = [h for h in headings if math.isfinite(h)]
        if len(headings) == 0:
            return 0.0

        return circular_mean(headings)

    def estimate_gyro_bias(self, average_seconds: float = 1.5) -> float:
        if self.df.empty:
            return 0.0

        t0 = float(self.df[self.timestamp_col].iloc[0])
        mask = self.df[self.timestamp_col] <= (t0 + average_seconds)
        vals = self.df.loc[mask, self.gyro_col].astype(float).tolist()

        vals = [v for v in vals if math.isfinite(v)]
        if len(vals) == 0:
            return 0.0

        return float(sum(vals) / len(vals))

    def step_events(self) -> Iterator[dict]:
        """
        Emit one event per detected step.
        heading_change is the accumulated gyro-integrated turn between steps.
        """
        turn_accum = 0.0

        for _, row in self.df.iterrows():
            dt = float(row[self.dt_col])
            ts = float(row[self.timestamp_col])
            step_flag = int(row[self.step_col])
            raw_heading = float(row[self.heading_col])
            gyro_z = float(row[self.gyro_col])  # expected rad/s

            if not math.isfinite(dt) or dt <= 0:
                continue
            if not math.isfinite(ts):
                continue
            if not math.isfinite(raw_heading):
                raw_heading = 0.0
            if not math.isfinite(gyro_z):
                gyro_z = 0.0

            # Integrate turn continuously between steps
            turn_accum += (gyro_z - self.gyro_bias) * dt

            if not math.isfinite(turn_accum):
                turn_accum = 0.0

            if step_flag == 1:
                heading_change = wrap_angle(turn_accum)

                if not math.isfinite(heading_change):
                    heading_change = 0.0

                yield {
                    "timestamp": ts,
                    "dt": dt,
                    "heading_change": heading_change,
                    "step_length_m": self.fixed_step_length_m,
                    "raw_heading": raw_heading,
                }

                turn_accum = 0.0


class LiveIMUReader:
    def __init__(self, sample_rate_hz: int = 50):
        if SenseHat is None:
            raise RuntimeError("sense_hat is not available on this machine.")
        self.sense = SenseHat()
        self.sample_rate_hz = sample_rate_hz

    def get_acceleration_norm(self) -> float:
        acc = self.sense.get_accelerometer_raw()
        ax = float(acc["x"])
        ay = float(acc["y"])
        az = float(acc["z"])
        return math.sqrt(ax * ax + ay * ay + az * az)

    def get_yaw(self) -> float:
        ori = self.sense.get_orientation_radians()
        yaw = float(ori["yaw"])
        if not math.isfinite(yaw):
            return 0.0
        return yaw

    def get_gyro_z(self) -> float:
        """
        Returns gyroscope z-axis angular velocity in rad/s.
        Sense HAT raw gyro often returns degrees/s, so convert.
        """
        gyro = self.sense.get_gyroscope_raw()
        gz = float(gyro["yaw"])
        if not math.isfinite(gz):
            return 0.0
        return math.radians(gz)

    def estimate_initial_heading(self, average_seconds: float = 1.5) -> float:
        headings = []
        start = time.time()
        dt_target = 1.0 / max(self.sample_rate_hz, 1)

        while (time.time() - start) < average_seconds:
            headings.append(self.get_yaw())
            time.sleep(dt_target)

        headings = [h for h in headings if math.isfinite(h)]
        if len(headings) == 0:
            return 0.0

        return circular_mean(headings)