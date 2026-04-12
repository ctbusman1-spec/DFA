from __future__ import annotations

"""
Floorplan utilities for the DFA IMU assignment.

This module provides:
- a lightweight static floorplan prior,
- coordinate conversion between meters and grid cells,
- a walkability prior function for the Bayesian filter,
- a trajectory validation utility for notebook figures and analysis.

The geometry is intentionally simple and transparent. It is adapted from the
validated indoor layout already used in earlier experiments, where the ground-
truth trajectory stayed fully inside the walkable region.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from config import get_config

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None


@dataclass(frozen=True)
class FloorplanMetadata:
    cell_size_cm: float
    width_cells: int
    height_cells: int
    width_m: float
    height_m: float
    pad_m: float
    pad_cells: int
    mirror_vertically: bool
    right_corridor_drop_m: float
    start_m: Tuple[float, float]
    start_cell: Tuple[int, int]


@dataclass(frozen=True)
class WalkabilityReport:
    total_samples: int
    inside_walkable: int
    outside_walkable: int
    walkability_percent: float


class FloorplanBuilder:
    """Construct a static floorplan prior and support coordinate queries."""

    def __init__(
        self,
        cell_size_cm: Optional[float] = None,
        sigma_cells: Optional[float] = None,
        draw_value: Optional[float] = None,
        max_prob_cap: Optional[float] = None,
        padding_m: Optional[float] = None,
        mirror_vertically: Optional[bool] = None,
        right_corridor_drop_m: Optional[float] = None,
    ) -> None:
        cfg = get_config()
        fp = cfg.floorplan
        self.cell_size_cm = fp.cell_size_cm if cell_size_cm is None else float(cell_size_cm)
        self.sigma_cells = fp.sigma_cells if sigma_cells is None else float(sigma_cells)
        self.draw_value = fp.draw_value if draw_value is None else float(draw_value)
        self.max_prob_cap = fp.max_prob_cap if max_prob_cap is None else float(max_prob_cap)
        self.padding_m = fp.padding_m if padding_m is None else float(padding_m)
        self.mirror_vertically = fp.mirror_vertically if mirror_vertically is None else bool(mirror_vertically)
        self.right_corridor_drop_m = fp.right_corridor_drop_m if right_corridor_drop_m is None else float(right_corridor_drop_m)
        self.start_position_m = tuple(map(float, fp.start_position_m))

        self.prior: Optional[np.ndarray] = None
        self.meta: Optional[FloorplanMetadata] = None

    def m_to_cell(self, meters: float) -> int:
        return int(round((meters * 100.0) / self.cell_size_cm))

    @staticmethod
    def add_rect(grid: np.ndarray, x0: int, y0: int, w: int, h: int, val: float) -> None:
        H, W = grid.shape
        x1 = max(0, min(W, x0 + w))
        y1 = max(0, min(H, y0 + h))
        x0 = max(0, min(W, x0))
        y0 = max(0, min(H, y0))
        if x1 > x0 and y1 > y0:
            grid[y0:y1, x0:x1] = np.maximum(grid[y0:y1, x0:x1], val)

    def build_prior(self) -> Tuple[np.ndarray, FloorplanMetadata]:
        """Create the static floorplan density p(x|FP)."""
        if gaussian_filter is None:
            raise ImportError("scipy is required for floorplan smoothing. Please install scipy.")

        # Transparent hand-built geometry in meters.
        W_TOTAL = 7.0
        CORR_W = 0.8
        H_LEFT = 2.3
        H_EXTRA = 1.5
        H_TOTAL = H_LEFT + H_EXTRA

        X_JUNCTION = 2.0
        RIGHT_LEN = 5.0

        X_STEP = 1.4
        X_REMAIN = 0.6

        start_x, start_y = self.start_position_m

        pad_cells = self.m_to_cell(self.padding_m)
        W = self.m_to_cell(W_TOTAL + 2 * self.padding_m)
        H = self.m_to_cell(H_TOTAL + 2 * self.padding_m)
        floor = np.zeros((H, W), dtype=float)

        def to_grid(x_m: float, y_m: float) -> Tuple[int, int]:
            x = pad_cells + self.m_to_cell(x_m)
            y = pad_cells + self.m_to_cell(y_m)
            return x, y

        # Left main block.
        x0, y0 = to_grid(0.0, 0.0)
        self.add_rect(floor, x0, y0, self.m_to_cell(X_JUNCTION), self.m_to_cell(H_LEFT), self.draw_value)

        # Top strip.
        x0, y0 = to_grid(0.0, 0.0)
        self.add_rect(floor, x0, y0, self.m_to_cell(X_JUNCTION), self.m_to_cell(CORR_W), self.draw_value)

        # Right corridor.
        x0, y0 = to_grid(X_JUNCTION, self.right_corridor_drop_m)
        self.add_rect(floor, x0, y0, self.m_to_cell(RIGHT_LEN), self.m_to_cell(CORR_W), self.draw_value)

        # Vertical junction stem.
        stem_height = max(0.0, H_TOTAL - self.right_corridor_drop_m)
        x0, y0 = to_grid(X_JUNCTION - CORR_W, self.right_corridor_drop_m)
        self.add_rect(floor, x0, y0, self.m_to_cell(CORR_W), self.m_to_cell(stem_height), self.draw_value)

        # Bottom step inside left block.
        x0, y0 = to_grid(0.0, H_LEFT - CORR_W)
        self.add_rect(floor, x0, y0, self.m_to_cell(X_STEP), self.m_to_cell(CORR_W), self.draw_value)

        x0, y0 = to_grid(X_STEP, H_LEFT - CORR_W)
        self.add_rect(floor, x0, y0, self.m_to_cell(X_REMAIN), self.m_to_cell(CORR_W), self.draw_value)

        smooth = gaussian_filter(floor, sigma=self.sigma_cells)
        if smooth.max() > 0:
            smooth = (smooth / smooth.max()) * self.max_prob_cap

        prior = smooth.copy()
        total = float(prior.sum())
        if total > 0:
            prior /= total

        sx, sy = to_grid(start_x, start_y)
        if self.mirror_vertically:
            prior = prior[::-1, :]
            sy = (H - 1) - sy

        meta = FloorplanMetadata(
            cell_size_cm=self.cell_size_cm,
            width_cells=W,
            height_cells=H,
            width_m=W * self.cell_size_cm / 100.0,
            height_m=H * self.cell_size_cm / 100.0,
            pad_m=self.padding_m,
            pad_cells=pad_cells,
            mirror_vertically=self.mirror_vertically,
            right_corridor_drop_m=self.right_corridor_drop_m,
            start_m=(start_x, start_y),
            start_cell=(sx, sy),
        )

        self.prior = prior
        self.meta = meta
        return prior.copy(), meta

    def ensure_prior(self) -> Tuple[np.ndarray, FloorplanMetadata]:
        if self.prior is None or self.meta is None:
            return self.build_prior()
        return self.prior.copy(), self.meta

    def meters_to_cell_xy(self, x_m: float, y_m: float) -> Tuple[int, int]:
        _, meta = self.ensure_prior()
        cx = meta.pad_cells + self.m_to_cell(float(x_m))
        cy = meta.pad_cells + self.m_to_cell(float(y_m))
        if meta.mirror_vertically:
            cy = (meta.height_cells - 1) - cy
        return int(cx), int(cy)

    def cell_to_probability(self, cx: int, cy: int) -> float:
        prior, meta = self.ensure_prior()
        if 0 <= cx < meta.width_cells and 0 <= cy < meta.height_cells:
            return float(prior[int(cy), int(cx)])
        return 0.0

    def prior_probability_at_meters(self, position_xyz: Iterable[float]) -> float:
        pos = np.asarray(position_xyz, dtype=float).reshape(-1)
        x_m = float(pos[0])
        y_m = float(pos[1])
        cx, cy = self.meters_to_cell_xy(x_m, y_m)
        return self.cell_to_probability(cx, cy)

    def is_walkable_meters(self, x_m: float, y_m: float) -> bool:
        return self.prior_probability_at_meters([x_m, y_m, 0.0]) > 0.0


class FloorplanValidator:
    """Validate trajectories against the static floorplan prior."""

    def __init__(self, builder: Optional[FloorplanBuilder] = None) -> None:
        self.builder = FloorplanBuilder() if builder is None else builder
        self.prior, self.meta = self.builder.ensure_prior()

    def validate_xy_trajectory(self, xs_m: Iterable[float], ys_m: Iterable[float]) -> Tuple[np.ndarray, WalkabilityReport]:
        xs = np.asarray(list(xs_m), dtype=float)
        ys = np.asarray(list(ys_m), dtype=float)
        if xs.shape != ys.shape:
            raise ValueError("X and Y trajectories must have the same shape.")

        valid = []
        for x, y in zip(xs, ys):
            if not np.isfinite(x) or not np.isfinite(y):
                valid.append(False)
                continue
            cx, cy = self.builder.meters_to_cell_xy(float(x), float(y))
            ok = 0 <= cx < self.meta.width_cells and 0 <= cy < self.meta.height_cells and self.prior[int(cy), int(cx)] > 0
            valid.append(bool(ok))

        valid_arr = np.asarray(valid, dtype=bool)
        inside = int(valid_arr.sum())
        total = int(len(valid_arr))
        outside = total - inside
        pct = 100.0 * inside / max(total, 1)
        report = WalkabilityReport(total, inside, outside, pct)
        return valid_arr, report

    def report_dict(self, report: WalkabilityReport) -> Dict[str, float | int]:
        return {
            "total_samples": report.total_samples,
            "inside_walkable": report.inside_walkable,
            "outside_walkable": report.outside_walkable,
            "walkability_percent": round(float(report.walkability_percent), 4),
        }


if __name__ == "__main__":
    from data_loader import DataLoader

    loader = DataLoader()
    demo_candidates = [loader.cfg.paths.data_file, Path("/mnt/data/Real-data-set-sensor_log_4.csv")]
    dataset_path = next((Path(p) for p in demo_candidates if Path(p).exists()), None)
    if dataset_path is None:
        raise FileNotFoundError("Could not find the benchmark dataset for the floorplan self-test.")

    df, info = loader.load_csv(dataset_path)
    builder = FloorplanBuilder()
    prior, meta = builder.build_prior()

    print("Floorplan dimensions:", f"{meta.width_cells}x{meta.height_cells} cells")
    print("Floorplan size (m):", round(meta.width_m, 4), "x", round(meta.height_m, 4))
    print("Start cell:", meta.start_cell)

    if info.has_ground_truth:
        validator = FloorplanValidator(builder)
        valid, report = validator.validate_xy_trajectory(df["gt_pos_x_m"], df["gt_pos_y_m"])
        print("Walkability report:")
        for key, value in validator.report_dict(report).items():
            print(f"- {key}: {value}")
    else:
        print("No ground-truth trajectory available, so walkability validation was skipped.")
