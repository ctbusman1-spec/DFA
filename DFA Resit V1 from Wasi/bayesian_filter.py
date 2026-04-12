from __future__ import annotations

"""
Non-recursive Bayesian filter with mode-seeking for the DFA IMU assignment.

This implementation follows the assignment logic rather than copying the paper
verbatim:
- a motion prior predicts where the pedestrian is likely to be next,
- a measurement likelihood scores agreement with an external position estimate,
- optional static priors can reward walkable regions,
- the posterior is approximated from sampled candidate positions,
- a mean-shift style mode-seeking step returns the most likely state.

The filter is designed to stay lightweight enough for Raspberry Pi style
constraints while remaining mathematically transparent for the notebook.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np

from config import get_config


@dataclass(frozen=True)
class BayesianStatistics:
    mode_confidence: float
    candidate_count: int
    samples_processed: int


class BayesianFilter:
    """Approximate non-recursive Bayesian filter with mode-seeking."""

    def __init__(
        self,
        process_noise: Optional[float] = None,
        measurement_noise: Optional[float] = None,
        window_size: Optional[int] = None,
        grid_resolution: Optional[float] = None,
        candidate_count: Optional[int] = None,
        mode_bandwidth_m: Optional[float] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        cfg = get_config()
        self.process_noise = cfg.bayesian.process_noise if process_noise is None else float(process_noise)
        self.measurement_noise = cfg.bayesian.measurement_noise if measurement_noise is None else float(measurement_noise)
        self.window_size = cfg.bayesian.window_size if window_size is None else int(window_size)
        self.grid_resolution = cfg.bayesian.grid_resolution_m if grid_resolution is None else float(grid_resolution)
        self.candidate_count = cfg.bayesian.candidate_count if candidate_count is None else int(candidate_count)
        self.mode_bandwidth_m = cfg.bayesian.mode_bandwidth_m if mode_bandwidth_m is None else float(mode_bandwidth_m)
        seed = cfg.evaluation.random_seed if random_seed is None else int(random_seed)
        self.rng = np.random.default_rng(seed)

        self.position = np.zeros(3, dtype=float)
        self.velocity = np.zeros(3, dtype=float)
        self.posterior_mean = np.zeros(3, dtype=float)
        self.posterior_cov = np.eye(3, dtype=float) * 0.1
        self.mode_position = np.zeros(3, dtype=float)
        self.mode_confidence = 0.0
        self.samples_processed = 0

        self.static_prior_fn: Optional[Callable[[np.ndarray], float]] = None

    def initialize_state(self, position: Iterable[float], velocity: Optional[Iterable[float]] = None) -> None:
        self.position = np.asarray(position, dtype=float).reshape(3)
        self.posterior_mean = self.position.copy()
        self.mode_position = self.position.copy()
        if velocity is None:
            self.velocity = np.zeros(3, dtype=float)
        else:
            self.velocity = np.asarray(velocity, dtype=float).reshape(3)

    def set_static_prior(self, prior_function: Optional[Callable[[np.ndarray], float]]) -> None:
        """Attach an optional walkability prior function p(x|FP)."""
        self.static_prior_fn = prior_function

    def predict_prior(self, accel_mps2: Iterable[float], dt: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        accel = np.asarray(accel_mps2, dtype=float).reshape(3)
        self.velocity = self.velocity + accel * dt
        self.position = self.position + self.velocity * dt + 0.5 * accel * dt**2

        prior_mean = self.position.copy()
        prior_cov = self.posterior_cov + np.eye(3, dtype=float) * self.process_noise
        return prior_mean, prior_cov

    @staticmethod
    def _gaussian_kernel(diff: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Unnormalized Gaussian density, vectorized over diff rows."""
        diff = np.atleast_2d(diff)
        cov = np.asarray(cov, dtype=float)
        cov = 0.5 * (cov + cov.T) + np.eye(cov.shape[0]) * 1e-9
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)
        quad = np.einsum("ni,ij,nj->n", diff, inv_cov, diff)
        return np.exp(-0.5 * quad)

    @staticmethod
    def _stride_ring_prior(positions: np.ndarray, previous_mean: np.ndarray, stride_length_m: float, sigma_m: float = 0.20) -> np.ndarray:
        radial_distance = np.linalg.norm(positions[:, :2] - previous_mean[None, :2], axis=1)
        error = radial_distance - float(stride_length_m)
        return np.exp(-0.5 * (error / sigma_m) ** 2)

    def _static_prior_vector(self, positions: np.ndarray) -> np.ndarray:
        if self.static_prior_fn is None:
            return np.ones(len(positions), dtype=float)
        values = np.array([max(float(self.static_prior_fn(pos)), 1e-12) for pos in positions], dtype=float)
        return values

    def generate_candidates(self, center: np.ndarray, spread_cov: np.ndarray, n_candidates: int) -> np.ndarray:
        sigma = max(np.sqrt(float(np.mean(np.diag(spread_cov)))), self.grid_resolution)
        return center[None, :] + sigma * self.rng.standard_normal(size=(n_candidates, 3))

    def mode_seeking(self, positions: np.ndarray, weights: np.ndarray, bandwidth: Optional[float] = None) -> Tuple[np.ndarray, float]:
        if len(positions) == 0:
            return self.posterior_mean.copy(), 0.0

        bandwidth = self.mode_bandwidth_m if bandwidth is None else float(bandwidth)
        safe_weights = np.maximum(weights, 1e-12)
        safe_weights = safe_weights / safe_weights.sum()
        mode_pos = np.average(positions, axis=0, weights=safe_weights)

        for _ in range(20):
            mode_old = mode_pos.copy()
            distances = np.linalg.norm(positions - mode_pos[None, :], axis=1)
            kernel_weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
            combined = kernel_weights * safe_weights
            denom = combined.sum()
            if denom <= 0:
                break
            mode_pos = (combined[:, None] * positions).sum(axis=0) / denom
            if np.linalg.norm(mode_pos - mode_old) < 1e-6:
                break

        return mode_pos, float(safe_weights.max())

    def update(
        self,
        measurement: Iterable[float],
        prior_mean: np.ndarray,
        prior_cov: np.ndarray,
        measurement_cov: Optional[np.ndarray] = None,
        stride_length_m: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Approximate Bayes update:
        p(x|z) ∝ p(z|x) p(x|motion) p(x|static) p(x|stride)
        """
        measurement = np.asarray(measurement, dtype=float).reshape(3)
        if measurement_cov is None:
            measurement_cov = np.eye(3, dtype=float) * self.measurement_noise

        n_meas = max(1, int(0.6 * self.candidate_count))
        n_prior = max(1, self.candidate_count - n_meas)
        meas_cov = np.asarray(measurement_cov, dtype=float) + np.eye(3) * 1e-6
        positions_meas = self.generate_candidates(measurement, meas_cov, n_meas)
        positions_prior = self.generate_candidates(prior_mean, prior_cov, n_prior)
        positions = np.vstack([positions_meas, positions_prior])
        likelihood_term = self._gaussian_kernel(measurement[None, :] - positions, meas_cov)
        prior_term = self._gaussian_kernel(positions - prior_mean[None, :], prior_cov)
        static_term = self._static_prior_vector(positions)
        if stride_length_m is None:
            stride_term = np.ones(len(positions), dtype=float)
        else:
            stride_term = self._stride_ring_prior(positions, self.posterior_mean, stride_length_m)

        posteriors = likelihood_term * prior_term * static_term * stride_term
        weight_sum = float(posteriors.sum())
        if weight_sum <= 0 or not np.isfinite(weight_sum):
            weights = np.ones(len(posteriors), dtype=float) / len(posteriors)
        else:
            weights = posteriors / weight_sum

        posterior_mean = np.average(positions, axis=0, weights=weights)
        centered = positions - posterior_mean[None, :]
        posterior_cov = (centered * weights[:, None]).T @ centered + np.eye(3, dtype=float) * 1e-6

        mode_pos, mode_conf = self.mode_seeking(positions, weights)

        self.posterior_mean = posterior_mean.copy()
        self.posterior_cov = posterior_cov.copy()
        self.mode_position = mode_pos.copy()
        self.mode_confidence = mode_conf
        self.position = mode_pos.copy()
        self.samples_processed += 1

        return self.mode_position.copy(), self.posterior_cov.copy()

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.mode_position.copy(), self.velocity.copy()

    def get_statistics(self) -> Dict[str, float | int]:
        stats = BayesianStatistics(
            mode_confidence=float(self.mode_confidence),
            candidate_count=int(self.candidate_count),
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
    bayes = BayesianFilter(candidate_count=400)
    detector = ZUPTDetector()

    if info.has_ground_truth:
        init_pos = [float(df["gt_pos_x_m"].iloc[0]), float(df["gt_pos_y_m"].iloc[0]), 0.0]
    else:
        init_pos = [0.0, 0.0, 0.0]

    kf.initialize_state(init_pos, [0.0, 0.0, 0.0])
    bayes.initialize_state(init_pos, [0.0, 0.0, 0.0])

    errors_kf = []
    errors_bayes = []

    previous_gt = None
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

        prior_mean, prior_cov = bayes.predict_prior(accel, row.dt_s)
        measurement_cov = np.eye(3, dtype=float) * 0.05

        pos_bayes, _ = bayes.update(pos_kf, prior_mean, prior_cov, measurement_cov, stride_length_m=None)

        if info.has_ground_truth and np.isfinite(row.gt_pos_x_m) and np.isfinite(row.gt_pos_y_m):
            gt_xy = np.array([row.gt_pos_x_m, row.gt_pos_y_m], dtype=float)
            errors_kf.append(float(np.linalg.norm(pos_kf[:2] - gt_xy)))
            errors_bayes.append(float(np.linalg.norm(pos_bayes[:2] - gt_xy)))

    print("Loaded dataset:", info.source_name)
    print("Dataset type:", info.dataset_type)
    print("Samples:", len(df))
    print("Bayesian statistics:")
    for key, value in bayes.get_statistics().items():
        print(f"- {key}: {value}")

    if errors_bayes:
        kf_arr = np.asarray(errors_kf, dtype=float)
        bayes_arr = np.asarray(errors_bayes, dtype=float)
        print("Linear KF 2D MAE (m):", round(float(kf_arr.mean()), 4))
        print("Bayesian 2D MAE (m):", round(float(bayes_arr.mean()), 4))
        print("Linear KF 2D RMSE (m):", round(float(np.sqrt(np.mean(kf_arr**2))), 4))
        print("Bayesian 2D RMSE (m):", round(float(np.sqrt(np.mean(bayes_arr**2))), 4))
        improvement = (float(kf_arr.mean()) - float(bayes_arr.mean())) / max(float(kf_arr.mean()), 1e-9) * 100.0
        print("Improvement vs Linear KF (%):", round(improvement, 2))
