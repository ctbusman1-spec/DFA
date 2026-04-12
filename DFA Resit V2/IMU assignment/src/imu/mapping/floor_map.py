from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np


@dataclass
class FloorMap:
    probability_map: np.ndarray
    width_m: float
    height_m: float
    scale_m_per_cell: float
    origin_xy_m: tuple[float, float] = (0.0, 0.0)
    walkable_map: np.ndarray | None = None
    polygon_xy_m: np.ndarray | None = None
    metadata: dict | None = None

    def __post_init__(self):
        self.probability_map = np.asarray(self.probability_map, dtype=float)
        self.width_m = float(self.width_m)
        self.height_m = float(self.height_m)
        self.scale_m_per_cell = float(self.scale_m_per_cell)
        self.origin_x_m = float(self.origin_xy_m[0])
        self.origin_y_m = float(self.origin_xy_m[1])
        self.ny, self.nx = self.probability_map.shape
        if self.walkable_map is not None:
            self.walkable_map = np.asarray(self.walkable_map, dtype=float)

    @classmethod
    def from_pickle(cls, file_path: str | Path) -> "FloorMap":
        file_path = Path(file_path)
        with file_path.open("rb") as f:
            data = pickle.load(f)
        return cls(
            probability_map=data["probability_map"],
            width_m=data["width_m"],
            height_m=data["height_m"],
            scale_m_per_cell=data["scale_m_per_cell"],
            origin_xy_m=tuple(data.get("origin_xy_m", (0.0, 0.0))),
            walkable_map=data.get("walkable_map"),
            polygon_xy_m=data.get("polygon_xy_m"),
            metadata={k: v for k, v in data.items() if k not in {"probability_map", "walkable_map", "polygon_xy_m", "width_m", "height_m", "scale_m_per_cell", "origin_xy_m"}},
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.probability_map.shape

    @property
    def extent_xy(self) -> tuple[float, float, float, float]:
        x0 = self.origin_x_m
        x1 = self.origin_x_m + self.width_m
        y0 = self.origin_y_m
        y1 = self.origin_y_m + self.height_m
        return x0, x1, y0, y1

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

    def is_walkable(self, x_m: float, y_m: float, threshold: float = 0.5) -> bool:
        if not self.in_bounds(x_m, y_m):
            return False
        if self.walkable_map is None:
            return self.get_probability(x_m, y_m) > 0.0
        ix, iy = self.xy_to_cell(x_m, y_m)
        return bool(self.walkable_map[iy, ix] >= threshold)
