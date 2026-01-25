"""
Linear Kalman Filter for Pedestrian Inertial Navigation
========================================================

Implements a simple 6-state Kalman filter for sensor fusion:
States: [x, y, z, vx, vy, vz] (position + velocity)

Optimized for Raspberry Pi:
- O(n³) complexity but small state space (6x6 matrices)
- Minimal memory footprint
- No matrix inversions needed (analytical solution)

Reference: Welch & Bishop (2006) "An Introduction to the Kalman Filter"
Jiménez et al. (2010) "Indoor Navigation System Using IMU"

Author: IMU Assignment Part 2
Date: 2026-01-25
"""

import numpy as np
import warnings


class LinearKalmanFilter:
    """
    6-State Linear Kalman Filter for Pedestrian Navigation
    
    States: [x, y, z, vx, vy, vz]
    """
    
    def __init__(self,
                 dt=0.05,
                 process_noise_accel=0.1,
                 process_noise_drift=0.01,
                 measurement_noise_accel=0.1,
                 measurement_noise_velocity=0.05,
                 measurement_noise_zupt=0.01):
        
        self.dt = dt
        
        # State: [x, y, z, vx, vy, vz]
        self.x_hat = np.zeros(6)
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        
        # Process noise covariance
        q_accel = process_noise_accel
        q_drift = process_noise_drift
        self.Q = np.diag([
            q_accel * dt**4 / 4, q_accel * dt**4 / 4, q_accel * dt**4 / 4,
            q_drift * dt**2 / 2, q_drift * dt**2 / 2, q_drift * dt**2 / 2,
        ]).astype(np.float32)
        
        # Measurement noise covariances
        self.R_accel = np.diag([measurement_noise_accel**2] * 3).astype(np.float32)
        self.R_velocity = np.diag([measurement_noise_velocity**2] * 3).astype(np.float32)
        self.R_zupt = np.diag([measurement_noise_zupt**2] * 3).astype(np.float32)
        
        # Initial error covariance
        self.P = np.eye(6) * 1.0
        
        # Measurement matrices
        self.H_accel = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        
        self.H_velocity = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        
        self.H_zupt = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        
        self.innovation_accel = np.zeros(3)
    
    def predict(self):
        """Prediction step (time update)."""
        self.x_hat = self.F @ self.x_hat
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.P = (self.P + self.P.T) / 2
    
    def update_accelerometer(self, accel_measurement):
        """Update with accelerometer measurement."""
        accel = np.asarray(accel_measurement, dtype=np.float32)
        z = accel * self.dt
        self._kalman_update(z, self.H_accel, self.R_accel)
    
    def update_velocity(self, velocity_measurement):
        """Update with direct velocity measurement."""
        z = np.asarray(velocity_measurement, dtype=np.float32)
        self._kalman_update(z, self.H_velocity, self.R_velocity)
    
    def update_zupt(self, measurement_noise=None):
        """Zero Velocity Update (ZUPT)."""
        z = np.zeros(3)
        R = self.R_zupt if measurement_noise is None else np.eye(3) * measurement_noise**2
        self._kalman_update(z, self.H_zupt, R)
    
    def _kalman_update(self, z, H, R):
        """Generic Kalman filter update step."""
        y = z - H @ self.x_hat
        S = H @ self.P @ H.T + R
        S = (S + S.T) / 2
        
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x_hat = self.x_hat + K @ y
            
            n = self.P.shape[0]
            I_KH = np.eye(n) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
            self.P = (self.P + self.P.T) / 2
            self.innovation_accel = y[:3] if H.shape[0] == 3 else self.innovation_accel
            
        except np.linalg.LinAlgError:
            warnings.warn("Singular matrix in Kalman update - skipping update")
    
    def get_state(self):
        """Get current state estimate."""
        return self.x_hat[:3], self.x_hat[3:]
    
    def get_covariance(self):
        """Get error covariance matrix."""
        return self.P.copy()
    
    def get_uncertainty(self):
        """Get position and velocity uncertainty."""
        position_std = np.sqrt(np.diag(self.P)[:3])
        velocity_std = np.sqrt(np.diag(self.P)[3:])
        return position_std, velocity_std
    
    def initialize_state(self, position, velocity=None):
        """Manually set initial state estimate."""
        self.x_hat[:3] = np.asarray(position, dtype=np.float32)
        if velocity is not None:
            self.x_hat[3:] = np.asarray(velocity, dtype=np.float32)
    
    def reset(self):
        """Reset filter to initial state."""
        self.x_hat = np.zeros(6)
        self.P = np.eye(6) * 1.0
