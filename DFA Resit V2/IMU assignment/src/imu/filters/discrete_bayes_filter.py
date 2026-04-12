from __future__ import annotations

import numpy as np


class DiscreteBayesFilter:
    def __init__(self, floor_map, config: dict, motion_cfg: dict, initial_state: dict):
        self.floor_map = floor_map
        self.cfg = config
        self.motion_cfg = motion_cfg

        self.grid_res = float(config["grid_resolution_xy_m"])
        self.heading_bins = int(config["heading_bins"])
        self.transition_spread_cells = int(config.get("transition_spread_cells", 1))
        self.use_stride_ring = bool(config.get("use_stride_ring", True))
        self.stride_ring_sigma_m = float(config.get("stride_ring_sigma_m", 0.18))

        self.directional_persistence = float(motion_cfg["directional_persistence"])
        self.forward_bias_std = float(motion_cfg["forward_bias_std_rad"])
        self.turn_gain = float(motion_cfg.get("turn_gain", 1.0))

        x0 = self.floor_map.origin_x_m
        x1 = self.floor_map.origin_x_m + self.floor_map.width_m
        y0 = self.floor_map.origin_y_m
        y1 = self.floor_map.origin_y_m + self.floor_map.height_m

        self.xs = np.arange(x0 + 0.5 * self.grid_res, x1, self.grid_res)
        self.ys = np.arange(y0 + 0.5 * self.grid_res, y1, self.grid_res)
        self.hs = np.linspace(-np.pi, np.pi, self.heading_bins, endpoint=False)

        self.belief = np.zeros((len(self.ys), len(self.xs), len(self.hs)), dtype=float)
        self._map_likelihood = np.zeros((len(self.ys), len(self.xs)), dtype=float)
        self._grid_x, self._grid_y = np.meshgrid(self.xs, self.ys)

        for iy, y in enumerate(self.ys):
            for ix, x in enumerate(self.xs):
                self._map_likelihood[iy, ix] = self.floor_map.get_probability(x, y)

        x_init, y_init = initial_state["position"]
        h_init = initial_state["heading"]

        ix0 = int(np.argmin(np.abs(self.xs - x_init)))
        iy0 = int(np.argmin(np.abs(self.ys - y_init)))
        ih0 = int(np.argmin(np.abs(self._angle_diff(self.hs, h_init))))

        self.belief[iy0, ix0, ih0] = 1.0
        self._normalize()

    def update_step(self, heading_change: float, step_length_m: float, dt: float = 1.0) -> dict:
        predicted = np.zeros_like(self.belief)
        spread_offsets = np.arange(-self.transition_spread_cells, self.transition_spread_cells + 1)
        step_kernel = np.exp(-0.5 * (spread_offsets / max(self.transition_spread_cells, 1)) ** 2)
        step_kernel /= step_kernel.sum()

        prev_mean, _, _ = self.estimate()
        prior_nonzero = np.argwhere(self.belief > 1.0e-12)

        for iy, ix, ih in prior_nonzero:
            p = self.belief[iy, ix, ih]
            persistence_pull = np.random.normal(0.0, self.forward_bias_std)
            effective_turn = self.turn_gain * (
                self.directional_persistence * heading_change
                + (1.0 - self.directional_persistence) * persistence_pull
            )
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

        if self.use_stride_ring:
            dist = np.sqrt((self._grid_x - prev_mean[0]) ** 2 + (self._grid_y - prev_mean[1]) ** 2)
            ring = np.exp(-0.5 * ((dist - step_length_m) / max(self.stride_ring_sigma_m, 1.0e-6)) ** 2)
            predicted *= ring[:, :, None]

        predicted *= self._map_likelihood[:, :, None]
        self.belief = predicted
        self._normalize()

        mean, var, mean_heading = self.estimate()
        return {
            "position": mean,
            "variance": var,
            "heading_rad": mean_heading,
            "neff": float(np.count_nonzero(self.belief > 1.0e-12)),
        }

    def estimate(self):
        pxy = self.belief.sum(axis=2)
        mean_x = np.sum(self._grid_x * pxy)
        mean_y = np.sum(self._grid_y * pxy)

        var_x = np.sum(((self._grid_x - mean_x) ** 2) * pxy)
        var_y = np.sum(((self._grid_y - mean_y) ** 2) * pxy)

        ph = self.belief.sum(axis=(0, 1))
        mean_heading = float(np.arctan2(np.sum(np.sin(self.hs) * ph), np.sum(np.cos(self.hs) * ph)))

        return np.array([mean_x, mean_y]), np.array([var_x, var_y]), mean_heading

    def occupancy_map(self, shape):
        occ = np.zeros(shape, dtype=float)
        pxy = self.belief.sum(axis=2)
        for iy, y in enumerate(self.ys):
            for ix, x in enumerate(self.xs):
                if self.floor_map.in_bounds(x, y):
                    cell_x, cell_y = self.floor_map.xy_to_cell(x, y)
                    if 0 <= cell_x < shape[1] and 0 <= cell_y < shape[0]:
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
