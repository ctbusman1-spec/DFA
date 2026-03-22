from __future__ import annotations

import numpy as np


class DiscreteBayesFilter:
    def __init__(self, floor_map, config: dict, motion_cfg: dict, initial_state: dict):
        self.floor_map = floor_map
        self.cfg = config
        self.motion_cfg = motion_cfg
        self.dx = float(config["grid_resolution_xy_m"])
        self.dy = self.dx
        self.nh = int(config["heading_bins"])
        self.top_k = int(config.get("top_k_transitions", 7))
        self.transition_spread = int(config.get("transition_spread_cells", 1))
        self.directional_persistence = float(motion_cfg["directional_persistence"])
        self.heading_noise_std = float(motion_cfg["heading_noise_std_rad"])
        self.forward_bias_std = float(motion_cfg["forward_bias_std_rad"])
        self.step_length_std = float(motion_cfg["step_length_std_m"])

        self.xs = np.arange(0.0, floor_map.width_m + self.dx, self.dx)
        self.ys = np.arange(0.0, floor_map.height_m + self.dy, self.dy)
        self.hs = np.linspace(-np.pi, np.pi, self.nh, endpoint=False)
        self.belief = np.zeros((len(self.ys), len(self.xs), self.nh), dtype=float)
        self._map_likelihood = self._precompute_map_likelihood()
        self.initialize_gaussian(initial_state)

    def _precompute_map_likelihood(self):
        out = np.zeros((len(self.ys), len(self.xs)), dtype=float)
        for iy, y in enumerate(self.ys):
            for ix, x in enumerate(self.xs):
                out[iy, ix] = self.floor_map.probability(x, y)
        out += 1e-15
        out /= out.sum()
        return out

    def initialize_gaussian(self, initial_state: dict):
        x0, y0 = initial_state["position"]
        h0 = initial_state["heading"]
        X, Y = np.meshgrid(self.xs, self.ys)
        spatial = np.exp(-0.5 * (((X - x0) / 0.45) ** 2 + ((Y - y0) / 0.45) ** 2))
        heading = np.exp(-0.5 * (self._angle_diff(self.hs, h0) / 0.25) ** 2)
        self.belief = spatial[:, :, None] * heading[None, None, :]
        self._normalize()

    def update_step(self, heading_change: float, step_length_m: float, dt: float = 1.0):
        predicted = np.zeros_like(self.belief)
        effective_turn = self.directional_persistence * heading_change
        step_kernel = np.array([0.2, 0.6, 0.2]) if self.transition_spread > 0 else np.array([1.0])
        spread_offsets = np.arange(-self.transition_spread, self.transition_spread + 1)

        active = np.argwhere(self.belief > 1e-10)
        for iy, ix, ih in active:
            p = self.belief[iy, ix, ih]
            if p <= 0.0:
                continue
            new_h = self._wrap_angle(self.hs[ih] + effective_turn)
            h_idx = int(np.argmin(np.abs(self._angle_diff(self.hs, new_h))))
            d = max(0.0, step_length_m)
            x_new = self.xs[ix] + d * np.cos(new_h)
            y_new = self.ys[iy] + d * np.sin(new_h)
            jx = int(np.argmin(np.abs(self.xs - x_new)))
            jy = int(np.argmin(np.abs(self.ys - y_new)))
            for off_x, kx in zip(spread_offsets, step_kernel):
                for off_y, ky in zip(spread_offsets, step_kernel):
                    tx = jx + off_x
                    ty = jy + off_y
                    if 0 <= tx < len(self.xs) and 0 <= ty < len(self.ys):
                        predicted[ty, tx, h_idx] += p * kx * ky

        predicted *= self._map_likelihood[:, :, None]
        self.belief = predicted
        self._normalize()
        mean, var = self.estimate()
        return {"position": mean, "variance": var, "neff": float(np.count_nonzero(self.belief > 1e-12))}

    def estimate(self):
        pxy = self.belief.sum(axis=2)
        X, Y = np.meshgrid(self.xs, self.ys)
        mean_x = np.sum(X * pxy)
        mean_y = np.sum(Y * pxy)
        var_x = np.sum(((X - mean_x) ** 2) * pxy)
        var_y = np.sum(((Y - mean_y) ** 2) * pxy)
        return np.array([mean_x, mean_y]), np.array([var_x, var_y])

    def occupancy_map(self, shape):
        occ = np.zeros(shape, dtype=float)
        pxy = self.belief.sum(axis=2)
        for iy, y in enumerate(self.ys):
            for ix, x in enumerate(self.xs):
                cell_x = min(shape[1] - 1, int(x / self.floor_map.scale_m_per_cell))
                cell_y = min(shape[0] - 1, int(y / self.floor_map.scale_m_per_cell))
                occ[cell_y, cell_x] += pxy[iy, ix]
        return occ

    def _normalize(self):
        total = self.belief.sum()
        if not np.isfinite(total) or total <= 0.0:
            self.belief[:] = 1.0 / self.belief.size
        else:
            self.belief /= total

    @staticmethod
    def _wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _angle_diff(a, b):
        return (a - b + np.pi) % (2 * np.pi) - np.pi
