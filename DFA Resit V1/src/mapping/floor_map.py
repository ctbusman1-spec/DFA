from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class FloorMap:
    pdf: np.ndarray
    scale_m_per_cell: float

    @property
    def shape(self) -> Tuple[int, int]:
        return self.pdf.shape

    @property
    def width_m(self) -> float:
        return self.pdf.shape[1] * self.scale_m_per_cell

    @property
    def height_m(self) -> float:
        return self.pdf.shape[0] * self.scale_m_per_cell

    @classmethod
    def create_football_field(
        cls,
        width_m: float = 20.0,
        height_m: float = 10.0,
        scale_m_per_cell: float = 0.1,
        sigma_cells: float = 2.5,
        edge_decay: float = 0.55,
        center_boost: float = 0.15,
    ) -> "FloorMap":
        h = max(2, int(round(height_m / scale_m_per_cell)))
        w = max(2, int(round(width_m / scale_m_per_cell)))

        raw = np.zeros((h, w), dtype=float)
        raw[:, :] = 1.0

        # Build a soft spatial distribution instead of a binary mask.
        yy, xx = np.mgrid[0:h, 0:w]
        xnorm = (xx - (w - 1) / 2.0) / max(1.0, w / 2.0)
        ynorm = (yy - (h - 1) / 2.0) / max(1.0, h / 2.0)

        # Slightly prefer central movement corridors.
        center_term = np.exp(-(ynorm**2) / 0.20)
        longitudinal_term = np.exp(-(xnorm**2) / 1.5)
        prior = raw * (1.0 - edge_decay * np.abs(ynorm))
        prior += center_boost * center_term * longitudinal_term

        # Make boundaries less likely but still valid inside the field.
        prior[:, 0] *= 0.35
        prior[:, -1] *= 0.35
        prior[0, :] *= 0.35
        prior[-1, :] *= 0.35

        pdf = gaussian_filter(prior, sigma=sigma_cells)
        pdf = np.clip(pdf, 0.0, None)
        pdf /= pdf.sum()
        return cls(pdf=pdf, scale_m_per_cell=scale_m_per_cell)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"pdf": self.pdf, "scale": self.scale_m_per_cell}, f)

    @classmethod
    def load(cls, path: str | Path) -> "FloorMap":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(pdf=data["pdf"], scale_m_per_cell=data["scale"])

    def meters_to_cell(self, x_m: float, y_m: float) -> Tuple[float, float]:
        return x_m / self.scale_m_per_cell, y_m / self.scale_m_per_cell

    def in_bounds(self, x_m: float, y_m: float) -> bool:
        x_cell, y_cell = self.meters_to_cell(x_m, y_m)
        return 0 <= x_cell < self.pdf.shape[1] and 0 <= y_cell < self.pdf.shape[0]

    def probability(self, x_m: float, y_m: float) -> float:
        x_cell, y_cell = self.meters_to_cell(x_m, y_m)
        if not (0 <= x_cell < self.pdf.shape[1] - 1 and 0 <= y_cell < self.pdf.shape[0] - 1):
            return 0.0

        x0 = int(np.floor(x_cell))
        y0 = int(np.floor(y_cell))
        dx = x_cell - x0
        dy = y_cell - y0

        p00 = self.pdf[y0, x0]
        p10 = self.pdf[y0, x0 + 1]
        p01 = self.pdf[y0 + 1, x0]
        p11 = self.pdf[y0 + 1, x0 + 1]

        p0 = p00 * (1.0 - dx) + p10 * dx
        p1 = p01 * (1.0 - dx) + p11 * dx
        return float(p0 * (1.0 - dy) + p1 * dy)

    def extent(self):
        return [0.0, self.width_m, 0.0, self.height_m]
