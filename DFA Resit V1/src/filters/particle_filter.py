from __future__ import annotations

import numpy as np


class ParticleFilter:
    def __init__(self, floor_map, config: dict, motion_cfg: dict, initial_state: dict, rng=None):
        self.floor_map = floor_map
        self.cfg = config
        self.motion_cfg = motion_cfg
        self.rng = np.random.default_rng(rng)
        self.N = int(config["n_particles"])
        self.resample_threshold_ratio = float(config.get("resample_threshold_ratio", 0.5))
        self.resample_method = config.get("resample_method", "systematic")
        self.directional_persistence = float(motion_cfg["directional_persistence"])
        self.forward_bias_std = float(motion_cfg["forward_bias_std_rad"])
        self.heading_noise_std = float(motion_cfg["heading_noise_std_rad"])
        self.step_length_std = float(motion_cfg["step_length_std_m"])
        self.impossible_state_penalty = float(motion_cfg.get("impossible_state_penalty", 0.0))

        self.particles = np.zeros((self.N, 3), dtype=float)
        self.weights = np.full(self.N, 1.0 / self.N, dtype=float)
        self.create_gaussian_particles(initial_state)

    def create_gaussian_particles(self, initial_state: dict):
        x0, y0 = initial_state["position"]
        h0 = initial_state["heading"]
        self.particles[:, 0] = self.rng.normal(x0, 0.35, size=self.N)
        self.particles[:, 1] = self.rng.normal(y0, 0.35, size=self.N)
        self.particles[:, 2] = self._wrap_angle(self.rng.normal(h0, 0.20, size=self.N))
        self._weight_from_map()

    def update_step(self, heading_change: float, step_length_m: float, dt: float = 1.0) -> dict:
        heading_noise = self.rng.normal(0.0, self.heading_noise_std, size=self.N)
        persistence_pull = self.rng.normal(0.0, self.forward_bias_std, size=self.N)
        effective_turn = self.directional_persistence * heading_change + (1.0 - self.directional_persistence) * persistence_pull
        self.particles[:, 2] = self._wrap_angle(self.particles[:, 2] + effective_turn + heading_noise)

        step_noise = self.rng.normal(0.0, self.step_length_std, size=self.N)
        step = np.maximum(0.0, step_length_m + step_noise)
        self.particles[:, 0] += step * np.cos(self.particles[:, 2])
        self.particles[:, 1] += step * np.sin(self.particles[:, 2])

        self._weight_from_map()
        neff_before = self.neff
        if neff_before < self.resample_threshold_ratio * self.N:
            self._resample()
        mean, var = self.estimate()
        return {
            "position": mean,
            "variance": var,
            "neff": float(self.neff),
        }

    def _weight_from_map(self):
        probs = np.array([self.floor_map.probability(x, y) for x, y in self.particles[:, :2]], dtype=float)
        probs = np.maximum(probs, self.impossible_state_penalty)
        probs += 1e-15
        self.weights *= probs
        total = self.weights.sum()
        if not np.isfinite(total) or total <= 0.0:
            self.weights[:] = 1.0 / self.N
        else:
            self.weights /= total

    @property
    def neff(self) -> float:
        return 1.0 / np.sum(self.weights ** 2)

    def estimate(self):
        mean_x = np.average(self.particles[:, 0], weights=self.weights)
        mean_y = np.average(self.particles[:, 1], weights=self.weights)
        var_x = np.average((self.particles[:, 0] - mean_x) ** 2, weights=self.weights)
        var_y = np.average((self.particles[:, 1] - mean_y) ** 2, weights=self.weights)
        return np.array([mean_x, mean_y]), np.array([var_x, var_y])

    def occupancy_map(self, shape):
        occ = np.zeros(shape, dtype=float)
        xs = np.clip((self.particles[:, 0] / self.floor_map.scale_m_per_cell).astype(int), 0, shape[1] - 1)
        ys = np.clip((self.particles[:, 1] / self.floor_map.scale_m_per_cell).astype(int), 0, shape[0] - 1)
        for x, y, w in zip(xs, ys, self.weights):
            occ[y, x] += w
        return occ

    def _resample(self):
        if self.resample_method != "systematic":
            positions = (self.rng.random(self.N) + np.arange(self.N)) / self.N
        else:
            positions = (self.rng.random() + np.arange(self.N)) / self.N
        cumulative_sum = np.cumsum(self.weights)
        indexes = np.searchsorted(cumulative_sum, positions)
        self.particles = self.particles[indexes]
        self.weights.fill(1.0 / self.N)

    @staticmethod
    def _wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
