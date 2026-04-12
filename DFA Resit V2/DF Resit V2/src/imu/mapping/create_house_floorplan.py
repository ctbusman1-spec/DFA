from __future__ import annotations


import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.ndimage import gaussian_filter

from utils.paths import PROJECT_ROOT


def normalize_pdf(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0.0, None)
    total = arr.sum()
    if total > 0:
        arr = arr / total
    return arr


def build_house_floorplan():
    pts = np.array([
        (4.00, 0.00),
        (4.85, 0.00),
        (4.85, 4.60),
        (4.65, 4.60),
        (4.65, 4.65),
        (5.05, 4.65),
        (5.05, 5.35),
        (5.65, 5.35),
        (5.65, 6.90),
        (3.95, 6.90),
        (3.95, 5.25),
        (2.15, 5.25),
        (2.15, 4.65),
        (3.85, 4.65),
        (3.85, 3.90),
        (4.00, 3.90),
        (4.00, 2.80),
        (2.35, 2.80),
        (2.35, 2.00),
        (4.00, 2.00),
    ], dtype=float)

    start_xy = (5.25, 5.85)
    scale_m_per_cell = 0.05
    pad_m = 0.20
    sigma_cells = 2.0

    min_x = pts[:, 0].min() - pad_m
    max_x = pts[:, 0].max() + pad_m
    min_y = pts[:, 1].min() - pad_m
    max_y = pts[:, 1].max() + pad_m

    width_m = max_x - min_x
    height_m = max_y - min_y

    nx = int(np.ceil(width_m / scale_m_per_cell))
    ny = int(np.ceil(height_m / scale_m_per_cell))

    xs = min_x + (np.arange(nx) + 0.5) * scale_m_per_cell
    ys = min_y + (np.arange(ny) + 0.5) * scale_m_per_cell
    X, Y = np.meshgrid(xs, ys)

    path = MplPath(pts)
    sample_points = np.column_stack([X.ravel(), Y.ravel()])
    inside = path.contains_points(sample_points).reshape(ny, nx)
    walkable_map = inside.astype(float)

    smoothed = gaussian_filter(walkable_map, sigma=sigma_cells)
    smoothed += 1e-12
    probability_map = normalize_pdf(smoothed)

    start_ix = int((start_xy[0] - min_x) / scale_m_per_cell)
    start_iy = int((start_xy[1] - min_y) / scale_m_per_cell)

    out = {
        "probability_map": probability_map,
        "walkable_map": walkable_map,
        "width_m": width_m,
        "height_m": height_m,
        "scale_m_per_cell": scale_m_per_cell,
        "origin_xy_m": (min_x, min_y),
        "start_xy_m": start_xy,
        "start_cell": (start_ix, start_iy),
        "polygon_xy_m": pts,
        "sigma_cells": sigma_cells,
        "pad_m": pad_m,
    }

    return out


def save_floorplan_outputs(out: dict):
    out_dir = PROJECT_ROOT / "data" / "floor_plans"
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = out_dir / "house_floorplan_pdf.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(out, f)

    min_x, min_y = out["origin_xy_m"]
    max_x = min_x + out["width_m"]
    max_y = min_y + out["height_m"]
    pts = out["polygon_xy_m"]
    start_xy = out["start_xy_m"]

    walkable_png = out_dir / "house_floorplan_walkable.png"
    pdf_png = out_dir / "house_floorplan_pdf.png"

    fig, ax = plt.subplots(figsize=(7, 9))
    ax.imshow(
        out["walkable_map"],
        origin="lower",
        extent=[min_x, max_x, min_y, max_y],
        aspect="equal",
        cmap="Greys",
    )
    ax.plot(pts[:, 0], pts[:, 1], "r-", linewidth=2)
    ax.scatter(start_xy[0], start_xy[1], s=80, marker="o")
    ax.set_title("Binary walkable map")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.tight_layout()
    plt.savefig(walkable_png, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 9))
    ax.imshow(
        out["probability_map"],
        origin="lower",
        extent=[min_x, max_x, min_y, max_y],
        aspect="equal",
    )
    ax.plot(pts[:, 0], pts[:, 1], "w-", linewidth=1.5)
    ax.scatter(start_xy[0], start_xy[1], s=80, marker="o")
    ax.set_title("Smoothed normalized floorplan PDF")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.tight_layout()
    plt.savefig(pdf_png, dpi=220)
    plt.close(fig)

    return pkl_path, walkable_png, pdf_png


def main():
    out = build_house_floorplan()
    pkl_path, walkable_png, pdf_png = save_floorplan_outputs(out)
    print(f"Saved floorplan pickle to: {pkl_path}")
    print(f"Saved walkable preview to: {walkable_png}")
    print(f"Saved PDF preview to: {pdf_png}")
    print(f"PDF sum: {out['probability_map'].sum():.6f}")
    print(f"Map shape: {out['probability_map'].shape}")


if __name__ == "__main__":
    main()
