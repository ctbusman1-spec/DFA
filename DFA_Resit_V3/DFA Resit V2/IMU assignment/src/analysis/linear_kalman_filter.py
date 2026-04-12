from __future__ import annotations

import warnings
import numpy as np


class LinearKalmanFilter:
    """
    Simple 6-state linear Kalman filter for analysis and sensor filtering.

    State vector:
        [x, y, z, vx, vy, vz]
    """

    def __init__(
        self,
        dt: float = 0.05,
        process_noise_accel: float = 0.1,
        process_noise_drift: float = 0.01,
        measurement_noise_accel: float = 0.1,
        measurement_noise_velocity: float = 0.05,
        measurement_noise_zupt: float = 0.01,
    ):
        self.dt = float(dt)
        self.x_hat = np.zeros(6, dtype=np.float32)

        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

        q_accel = process_noise_accel
        q_drift = process_noise_drift
        self.Q = np.diag([
            q_accel * dt**4 / 4, q_accel * dt**4 / 4, q_accel * dt**4 / 4,
            q_drift * dt**2 / 2, q_drift * dt**2 / 2, q_drift * dt**2 / 2,
        ]).astype(np.float32)

        self.R_accel = np.diag([measurement_noise_accel**2] * 3).astype(np.float32)
        self.R_velocity = np.diag([measurement_noise_velocity**2] * 3).astype(np.float32)
        self.R_zupt = np.diag([measurement_noise_zupt**2] * 3).astype(np.float32)

        self.P = np.eye(6, dtype=np.float32) * 1.0

        self.H_velocity = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

    def predict(self):
        self.x_hat = self.F @ self.x_hat
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.P = (self.P + self.P.T) / 2

    def update_velocity(self, velocity_measurement):
        z = np.asarray(velocity_measurement, dtype=np.float32)
        self._kalman_update(z, self.H_velocity, self.R_velocity)

    def update_accelerometer(self, accel_measurement):
        z = np.asarray(accel_measurement, dtype=np.float32) * self.dt
        self._kalman_update(z, self.H_velocity, self.R_accel)

    def update_zupt(self, measurement_noise=None):
        z = np.zeros(3, dtype=np.float32)
        R = self.R_zupt if measurement_noise is None else np.eye(3, dtype=np.float32) * measurement_noise**2
        self._kalman_update(z, self.H_velocity, R)

    def _kalman_update(self, z, H, R):
        y = z - H @ self.x_hat
        S = H @ self.P @ H.T + R
        S = (S + S.T) / 2
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x_hat = self.x_hat + K @ y
            n = self.P.shape[0]
            I_KH = np.eye(n, dtype=np.float32) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
            self.P = (self.P + self.P.T) / 2
        except np.linalg.LinAlgError:
            warnings.warn("Singular matrix in Kalman update; update skipped.")

    def get_state(self):
        return self.x_hat[:3].copy(), self.x_hat[3:].copy()

    def get_uncertainty(self):
        position_std = np.sqrt(np.diag(self.P)[:3])
        velocity_std = np.sqrt(np.diag(self.P)[3:])
        return position_std, velocity_std

    def initialize_state(self, position, velocity=None):
        self.x_hat[:3] = np.asarray(position, dtype=np.float32)
        if velocity is not None:
            self.x_hat[3:] = np.asarray(velocity, dtype=np.float32)

    def reset(self):
        self.x_hat[:] = 0.0
        self.P = np.eye(6, dtype=np.float32) * 1.0
