from __future__ import annotations

"""
Particle filter for the DFA IMU assignment.

This implementation is designed as a lightweight comparison method against the
linear Kalman baseline and the Bayesian mode-seeking filter. It uses:
- a constant-acceleration particle prediction step,
- a measurement update driven by the linear KF position estimate,
- optional indoor floor-plane regularization,
- systematic resampling based on effective sample size.

The filter is intentionally pragmatic rather than fully physical. That makes it
fast enough for coursework experiments while still exposing the core trade-off:
more flexible state estimation at higher computational cost.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from config import get_config


@dataclass(frozen=True)
class ParticleStatistics:
    n_particles: int
    effective_sample_size: float
    resampling_count: int
    samples_processed: int


class ParticleFilter:
    """Simple particle filter over position and velocity in 3D."""

    def __init__(
        self,
        n_particles: Optional[int] = None,
        process_noise: Optional[float] = None,
        measurement_noise: Optional[float] = None,
        resampling_threshold: Optional[float] = None,
        planar_only: bool = True,
        random_seed: Optional[int] = None,
    ) -> None:
        cfg = get_config()
        self.n_particles = cfg.particle.n_particles if n_particles is None else int(n_particles)
        self.process_noise = cfg.particle.process_noise if process_noise is None else float(process_noise)
        self.measurement_noise = cfg.particle.measurement_noise if measurement_noise is None else float(measurement_noise)
        self.resampling_threshold = cfg.particle.resampling_threshold if resampling_threshold is None else float(resampling_threshold)
        self.planar_only = bool(planar_only)
        seed = cfg.evaluation.random_seed if random_seed is None else int(random_seed)
        self.rng = np.random.default_rng(seed)
        self.floor_z_m = cfg.units.floor_z_m

        self.particles = np.zeros((self.n_particles, 3), dtype=float)
        self.velocities = np.zeros((self.n_particles, 3), dtype=float)
        self.weights = np.ones(self.n_particles, dtype=float) / self.n_particles
        self.position = np.zeros(3, dtype=float)
        self.velocity = np.zeros(3, dtype=float)
        self.resampling_count = 0
        self.effective_sample_size = float(self.n_particles)
        self.samples_processed = 0

    def initialize_state(self, position: Iterable[float], velocity: Optional[Iterable[float]] = None, spread_m: float = 0.02) -> None:
        pos = np.asarray(position, dtype=float).reshape(3)
        vel = np.zeros(3, dtype=float) if velocity is None else np.asarray(velocity, dtype=float).reshape(3)
        self.particles = pos[None, :] + self.rng.normal(0.0, spread_m, size=(self.n_particles, 3))
        self.velocities = vel[None, :] + self.rng.normal(0.0, spread_m * 0.5, size=(self.n_particles, 3))
        if self.planar_only:
            self.particles[:, 2] = self.floor_z_m + self.rng.normal(0.0, spread_m * 0.1, size=self.n_particles)
            self.velocities[:, 2] = 0.0
        self.weights.fill(1.0 / self.n_particles)
        self.position = np.average(self.particles, axis=0, weights=self.weights)
        self.velocity = np.average(self.velocities, axis=0, weights=self.weights)
        self.effective_sample_size = float(self.n_particles)
        self.resampling_count = 0
        self.samples_processed = 0

    def predict(self, accel_mps2: Iterable[float], dt: float = 0.05) -> None:
        accel = np.asarray(accel_mps2, dtype=float).reshape(3)
        if self.planar_only:
            accel = accel.copy()
            accel[2] = 0.0

        accel_scale = 0.08
        vel_noise = self.rng.normal(0.0, self.process_noise * 0.2, size=(self.n_particles, 3))
        pos_noise = self.rng.normal(0.0, self.process_noise * max(dt, 1e-6) * 0.5, size=(self.n_particles, 3))

        self.velocities = 0.92 * self.velocities + accel_scale * accel[None, :] * dt + vel_noise
        speed = np.linalg.norm(self.velocities[:, :2], axis=1, keepdims=True)
        speed_clip = np.maximum(1.0, speed / 1.5)
        self.velocities[:, :2] = self.velocities[:, :2] / speed_clip
        self.particles = self.particles + self.velocities * dt + 0.5 * accel_scale * accel[None, :] * (dt ** 2) + pos_noise

        if self.planar_only:
            self.particles[:, 2] = self.floor_z_m + self.rng.normal(0.0, self.process_noise * 0.05, size=self.n_particles)
            self.velocities[:, 2] *= 0.2

        self.samples_processed += 1

    @staticmethod
    def _gaussian_likelihood(diff: np.ndarray, cov: np.ndarray) -> np.ndarray:
        cov = np.asarray(cov, dtype=float)
        cov = 0.5 * (cov + cov.T) + np.eye(cov.shape[0]) * 1e-9
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)
        quad = np.einsum("ni,ij,nj->n", diff, inv_cov, diff)
        return np.exp(-0.5 * quad)

    def update(self, measurement: Iterable[float], measurement_cov: Optional[np.ndarray] = None) -> None:
        meas = np.asarray(measurement, dtype=float).reshape(3)
        if measurement_cov is None:
            measurement_cov = np.eye(3, dtype=float) * (self.measurement_noise ** 2)
        else:
            measurement_cov = np.asarray(measurement_cov, dtype=float)

        diff = meas[None, :] - self.particles
        likelihoods = self._gaussian_likelihood(diff, measurement_cov)
        self.weights *= np.maximum(likelihoods, 1e-16)
        weight_sum = float(self.weights.sum())
        if weight_sum <= 0 or not np.isfinite(weight_sum):
            self.weights.fill(1.0 / self.n_particles)
        else:
            self.weights /= weight_sum

        self.effective_sample_size = float(1.0 / np.sum(self.weights ** 2))
        if self.effective_sample_size < self.resampling_threshold * self.n_particles:
            self.resample()

    def resample(self) -> None:
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        positions = (self.rng.random() + np.arange(self.n_particles)) / self.n_particles
        idx = np.searchsorted(cumulative, positions)
        self.particles = self.particles[idx]
        self.velocities = self.velocities[idx]
        self.weights.fill(1.0 / self.n_particles)
        self.effective_sample_size = float(self.n_particles)
        self.resampling_count += 1

    def estimate_state(self, method: str = "weighted_mean") -> Tuple[np.ndarray, np.ndarray]:
        if method == "weighted_mean":
            self.position = np.average(self.particles, axis=0, weights=self.weights)
            self.velocity = np.average(self.velocities, axis=0, weights=self.weights)
        elif method == "max_weight":
            idx = int(np.argmax(self.weights))
            self.position = self.particles[idx].copy()
            self.velocity = self.velocities[idx].copy()
        else:
            raise ValueError(f"Unsupported estimation method: {method}")
        return self.position.copy(), self.velocity.copy()

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.position.copy(), self.velocity.copy()

    def get_statistics(self) -> Dict[str, float | int]:
        stats = ParticleStatistics(
            n_particles=int(self.n_particles),
            effective_sample_size=float(self.effective_sample_size),
            resampling_count=int(self.resampling_count),
            samples_processed=int(self.samples_processed),
        )
        return stats.__dict__.copy()


if __name__ == "__main__":
    from pathlib import Path

    from data_loader import DataLoader
    from linear_kalman_filter import LinearKalmanFilter
    from zupt_detector import ZUPTDetector

    loader = DataLoader()
    demo_candidates = [loader.cfg.paths.data_file, Path("/mnt/data/Real-data-set-sensor_log_4.csv")]
    dataset_path = next((Path(p) for p in demo_candidates if Path(p).exists()), None)
    if dataset_path is None:
        raise FileNotFoundError("Could not find the benchmark dataset for the self-test.")

    df, info = loader.load_csv(dataset_path)

    kf = LinearKalmanFilter(planar_only=True)
    pf = ParticleFilter(n_particles=500, planar_only=True)
    detector = ZUPTDetector()

    if info.has_ground_truth:
        init_pos = [float(df["gt_pos_x_m"].iloc[0]), float(df["gt_pos_y_m"].iloc[0]), 0.0]
    else:
        init_pos = [0.0, 0.0, 0.0]

    kf.initialize_state(init_pos, [0.0, 0.0, 0.0])
    pf.initialize_state(init_pos, [0.0, 0.0, 0.0])

    errors_kf = []
    errors_pf = []

    for row in df.itertuples(index=False):
        accel = np.array([row.acc_x_mps2, row.acc_y_mps2, row.acc_z_mps2], dtype=float)
        gyro = np.array([row.gyro_x_rads, row.gyro_y_rads, row.gyro_z_rads], dtype=float)

        kf.predict(row.dt_s)
        if np.max(np.abs(accel[:2])) > 0.1:
            kf.update_accelerometer(accel)

        is_stationary, _ = detector.update(accel, gyro, row.step_event)
        if is_stationary:
            kf.update_zero_velocity()
        kf.update_floor_constraint()
        pos_kf, _ = kf.get_state()

        pf.predict(accel, row.dt_s)
        measurement_cov = np.eye(3, dtype=float) * 0.03
        pf.update(pos_kf, measurement_cov)
        pos_pf, _ = pf.estimate_state(method="weighted_mean")

        if info.has_ground_truth and np.isfinite(row.gt_pos_x_m) and np.isfinite(row.gt_pos_y_m):
            gt_xy = np.array([row.gt_pos_x_m, row.gt_pos_y_m], dtype=float)
            errors_kf.append(float(np.linalg.norm(pos_kf[:2] - gt_xy)))
            errors_pf.append(float(np.linalg.norm(pos_pf[:2] - gt_xy)))

    print("Loaded dataset:", info.source_name)
    print("Dataset type:", info.dataset_type)
    print("Samples:", len(df))
    print("Particle statistics:")
    for key, value in pf.get_statistics().items():
        print(f"- {key}: {value}")

    if errors_pf:
        kf_arr = np.asarray(errors_kf, dtype=float)
        pf_arr = np.asarray(errors_pf, dtype=float)
        print("Linear KF 2D MAE (m):", round(float(kf_arr.mean()), 4))
        print("Particle 2D MAE (m):", round(float(pf_arr.mean()), 4))
        print("Linear KF 2D RMSE (m):", round(float(np.sqrt(np.mean(kf_arr**2))), 4))
        print("Particle 2D RMSE (m):", round(float(np.sqrt(np.mean(pf_arr**2))), 4))
        improvement = (float(kf_arr.mean()) - float(pf_arr.mean())) / max(float(kf_arr.mean()), 1e-9) * 100.0
        print("Improvement vs Linear KF (%):", round(improvement, 2))
