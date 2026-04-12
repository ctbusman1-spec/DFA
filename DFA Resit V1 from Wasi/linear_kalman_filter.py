from __future__ import annotations

"""
Pragmatic linear Kalman filter baseline for the DFA IMU assignment.

This filter is intentionally lightweight and stable. It does not attempt to be a
full foot-mounted INS/EKF mechanization. Instead, it provides a simple linear
state-space baseline that:
- tracks position and velocity,
- uses acceleration-derived pseudo-velocity observations,
- applies zero-velocity updates during detected stationary phases,
- keeps the motion on the indoor floor plane.

This is suitable for the course requirement of including at least one linear
Kalman filter in the analysis, while keeping the implementation explainable and
fast enough for Raspberry Pi style constraints.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from config import get_config


@dataclass(frozen=True)
class FilterStatistics:
    innovation_norm: float
    zupt_updates: int
    floor_updates: int
    samples_processed: int


class LinearKalmanFilter:
    """Six-state linear Kalman filter: [x, y, z, vx, vy, vz]."""

    def __init__(
        self,
        dt: Optional[float] = None,
        process_noise_accel: Optional[float] = None,
        process_noise_drift: Optional[float] = None,
        measurement_noise_accel: Optional[float] = None,
        measurement_noise_velocity: Optional[float] = None,
        measurement_noise_zupt: Optional[float] = None,
        floor_measurement_noise: float = 0.01,
        planar_only: bool = True,
    ) -> None:
        cfg = get_config()
        self.dt = cfg.kalman.dt_default_s if dt is None else float(dt)
        self.process_noise_accel = cfg.kalman.process_noise_accel if process_noise_accel is None else float(process_noise_accel)
        self.process_noise_drift = cfg.kalman.process_noise_drift if process_noise_drift is None else float(process_noise_drift)
        self.measurement_noise_accel = cfg.kalman.measurement_noise_accel if measurement_noise_accel is None else float(measurement_noise_accel)
        self.measurement_noise_velocity = cfg.kalman.measurement_noise_velocity if measurement_noise_velocity is None else float(measurement_noise_velocity)
        self.measurement_noise_zupt = cfg.kalman.measurement_noise_zupt if measurement_noise_zupt is None else float(measurement_noise_zupt)
        self.floor_measurement_noise = float(floor_measurement_noise)
        self.planar_only = bool(planar_only)
        self.floor_z_m = cfg.units.floor_z_m

        self.x_hat = np.zeros(6, dtype=float)
        self.P = np.eye(6, dtype=float) * 1.0
        self.innovation_accel = np.zeros(3, dtype=float)

        self.zupt_updates = 0
        self.floor_updates = 0
        self.samples_processed = 0

        self._set_matrices(self.dt)

    def _set_matrices(self, dt: float) -> None:
        self.dt = float(dt)
        self.F = np.array(
            [
                [1.0, 0.0, 0.0, dt, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, dt, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, dt],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        q_accel = self.process_noise_accel
        q_drift = self.process_noise_drift
        self.Q = np.diag(
            [
                q_accel * dt**4 / 4.0,
                q_accel * dt**4 / 4.0,
                (q_accel * dt**4 / 4.0) * (0.05 if self.planar_only else 1.0),
                q_drift * dt**2 / 2.0,
                q_drift * dt**2 / 2.0,
                (q_drift * dt**2 / 2.0) * (0.05 if self.planar_only else 1.0),
            ]
        )

        self.R_accel = np.diag([self.measurement_noise_accel**2] * 3)
        self.R_velocity = np.diag([self.measurement_noise_velocity**2] * 3)
        self.R_zupt = np.diag([self.measurement_noise_zupt**2] * 3)

        self.H_velocity = np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        self.H_floor_pos = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=float)
        self.H_floor_vel = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=float)

    def initialize_state(self, position: Iterable[float], velocity: Optional[Iterable[float]] = None) -> None:
        self.x_hat[:3] = np.asarray(position, dtype=float).reshape(3)
        if velocity is None:
            self.x_hat[3:] = 0.0
        else:
            self.x_hat[3:] = np.asarray(velocity, dtype=float).reshape(3)

    def predict(self, dt: Optional[float] = None) -> None:
        if dt is not None and np.isfinite(dt) and dt > 0:
            self._set_matrices(float(dt))
        self.x_hat = self.F @ self.x_hat
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)
        self.samples_processed += 1

    def update_accelerometer(self, accel_measurement_mps2: Iterable[float]) -> None:
        """
        Convert acceleration into a pseudo-velocity increment measurement.

        This is a pragmatic approximation used only for the linear KF baseline.
        It is deliberately lightweight and avoids the complexity of a full INS
        mechanization while still injecting sensor information into the filter.
        """
        accel = np.asarray(accel_measurement_mps2, dtype=float).reshape(3)
        if self.planar_only:
            accel = accel.copy()
            accel[2] = 0.0
        z = accel * self.dt
        self._kalman_update(z, self.H_velocity, self.R_accel)

    def update_velocity(self, velocity_measurement_mps: Iterable[float], measurement_noise: Optional[float] = None) -> None:
        z = np.asarray(velocity_measurement_mps, dtype=float).reshape(3)
        R = self.R_velocity if measurement_noise is None else np.eye(3) * float(measurement_noise) ** 2
        self._kalman_update(z, self.H_velocity, R)

    def update_zero_velocity(self, measurement_noise: Optional[float] = None) -> None:
        R = self.R_zupt if measurement_noise is None else np.eye(3) * float(measurement_noise) ** 2
        self._kalman_update(np.zeros(3, dtype=float), self.H_velocity, R)
        self.zupt_updates += 1

    def update_floor_constraint(self, z_value: Optional[float] = None, measurement_noise: Optional[float] = None) -> None:
        sigma = self.floor_measurement_noise if measurement_noise is None else float(measurement_noise)
        z_ref = self.floor_z_m if z_value is None else float(z_value)
        self._kalman_update(np.array([z_ref], dtype=float), self.H_floor_pos, np.array([[sigma**2]], dtype=float))
        self._kalman_update(np.array([0.0], dtype=float), self.H_floor_vel, np.array([[sigma**2]], dtype=float))
        self.floor_updates += 1

    def _kalman_update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        y = z - H @ self.x_hat
        S = H @ self.P @ H.T + R
        S = 0.5 * (S + S.T)
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        self.x_hat = self.x_hat + K @ y
        n = self.P.shape[0]
        I_KH = np.eye(n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        if y.size >= 3:
            self.innovation_accel = y[:3].copy()
        else:
            self.innovation_accel[: y.size] = y

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_hat[:3].copy(), self.x_hat[3:].copy()

    def get_covariance(self) -> np.ndarray:
        return self.P.copy()

    def get_uncertainty(self) -> Tuple[np.ndarray, np.ndarray]:
        std = np.sqrt(np.maximum(np.diag(self.P), 0.0))
        return std[:3].copy(), std[3:].copy()

    def get_statistics(self) -> Dict[str, float | int]:
        stats = FilterStatistics(
            innovation_norm=float(np.linalg.norm(self.innovation_accel)),
            zupt_updates=self.zupt_updates,
            floor_updates=self.floor_updates,
            samples_processed=self.samples_processed,
        )
        return stats.__dict__.copy()


if __name__ == "__main__":
    from pathlib import Path

    import numpy as np

    from data_loader import DataLoader
    from zupt_detector import ZUPTDetector

    loader = DataLoader()

    demo_candidates = [loader.cfg.paths.data_file, Path("/mnt/data/Real-data-set-sensor_log_4.csv")]
    dataset_path = next((Path(p) for p in demo_candidates if Path(p).exists()), None)
    if dataset_path is None:
        raise FileNotFoundError("Could not find the benchmark dataset for the self-test.")

    df, info = loader.load_csv(dataset_path)

    kf = LinearKalmanFilter(planar_only=True)
    detector = ZUPTDetector()

    if info.has_ground_truth:
        kf.initialize_state([float(df['gt_pos_x_m'].iloc[0]), float(df['gt_pos_y_m'].iloc[0]), 0.0], [0.0, 0.0, 0.0])
    else:
        kf.initialize_state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    positions = []
    errors_2d = []

    for row in df.itertuples(index=False):
        accel = np.array([row.acc_x_mps2, row.acc_y_mps2, row.acc_z_mps2], dtype=float)
        gyro = np.array([row.gyro_x_rads, row.gyro_y_rads, row.gyro_z_rads], dtype=float)

        kf.predict(row.dt_s)
        if np.max(np.abs(accel[:2])) > 0.1:
            kf.update_accelerometer(accel)

        is_stationary, confidence = detector.update(accel, gyro, row.step_event)
        if is_stationary:
            kf.update_zero_velocity()
        kf.update_floor_constraint()

        pos, vel = kf.get_state()
        positions.append(pos)

        if info.has_ground_truth and np.isfinite(row.gt_pos_x_m) and np.isfinite(row.gt_pos_y_m):
            gt = np.array([row.gt_pos_x_m, row.gt_pos_y_m], dtype=float)
            errors_2d.append(float(np.linalg.norm(pos[:2] - gt)))

    positions = np.asarray(positions, dtype=float)
    print("Loaded dataset:", info.source_name)
    print("Dataset type:", info.dataset_type)
    print("Samples:", len(df))
    print("Position range X (m):", round(float(positions[:, 0].min()), 4), "to", round(float(positions[:, 0].max()), 4))
    print("Position range Y (m):", round(float(positions[:, 1].min()), 4), "to", round(float(positions[:, 1].max()), 4))
    print("Filter statistics:")
    for key, value in kf.get_statistics().items():
        print(f"- {key}: {value}")

    if errors_2d:
        errors_arr = np.asarray(errors_2d, dtype=float)
        print("2D MAE (m):", round(float(errors_arr.mean()), 4))
        print("2D RMSE (m):", round(float(np.sqrt(np.mean(errors_arr**2))), 4))
        print("2D Max Error (m):", round(float(errors_arr.max()), 4))
