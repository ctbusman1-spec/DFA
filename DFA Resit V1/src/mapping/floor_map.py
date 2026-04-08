from __future__ import annotations

import pickle
import numpy as np


class FloorMap:
    def __init__(
        self,
        probability_map: np.ndarray,
        width_m: float,
        height_m: float,
        scale_m_per_cell: float,
        origin_xy_m=(0.0, 0.0),
    ):
        self.probability_map = probability_map.astype(float)
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.scale_m_per_cell = float(scale_m_per_cell)
        self.origin_x_m = float(origin_xy_m[0])
        self.origin_y_m = float(origin_xy_m[1])

        self.ny, self.nx = self.probability_map.shape

    @classmethod
    def from_pickle(cls, file_path: str) -> "FloorMap":
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        return cls(
            probability_map=data["probability_map"],
            width_m=data["width_m"],
            height_m=data["height_m"],
            scale_m_per_cell=data["scale_m_per_cell"],
            origin_xy_m=data.get("origin_xy_m", (0.0, 0.0)),
        )

    def xy_to_cell(self, x_m: float, y_m: float) -> tuple[int, int]:
        ix = int((x_m - self.origin_x_m) / self.scale_m_per_cell)
        iy = int((y_m - self.origin_y_m) / self.scale_m_per_cell)
        return ix, iy

    def in_bounds(self, x_m: float, y_m: float) -> bool:
        ix, iy = self.xy_to_cell(x_m, y_m)
        return 0 <= ix < self.nx and 0 <= iy < self.ny

    def get_probability(self, x_m: float, y_m: float) -> float:
        if not self.in_bounds(x_m, y_m):
            return 0.0

        ix, iy = self.xy_to_cell(x_m, y_m)
        return float(self.probability_map[iy, ix])