from __future__ import annotations

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter


def normalize_pdf(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0.0, None)
    s = arr.sum()
    if s > 0:
        arr = arr / s
    return arr


def main():
    # -------------------------------------------------
    # Jouw walkable contour in meters
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Grid settings
    # -------------------------------------------------
    scale_m_per_cell = 0.05   # 5 cm per cell
    pad_m = 0.20
    sigma_cells = 2.0         # smoothing for soft probability map

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

    # -------------------------------------------------
    # Rasterize polygon -> binary walkable map
    # -------------------------------------------------
    path = Path(pts)
    sample_points = np.column_stack([X.ravel(), Y.ravel()])
    inside = path.contains_points(sample_points).reshape(ny, nx)

    walkable = inside.astype(float)

    # -------------------------------------------------
    # Smooth -> probability map
    # -------------------------------------------------
    smoothed = gaussian_filter(walkable, sigma=sigma_cells)

    # tiny epsilon so we don't get all exact zeros outside
    smoothed += 1e-12

    probability_map = normalize_pdf(smoothed)

    # -------------------------------------------------
    # Convert start point to grid cell
    # -------------------------------------------------
    start_ix = int((start_xy[0] - min_x) / scale_m_per_cell)
    start_iy = int((start_xy[1] - min_y) / scale_m_per_cell)

    # -------------------------------------------------
    # Save
    # -------------------------------------------------
    out = {
        "probability_map": probability_map,
        "width_m": width_m,
        "height_m": height_m,
        "scale_m_per_cell": scale_m_per_cell,
        "origin_xy_m": (min_x, min_y),
        "start_xy_m": start_xy,
        "start_cell": (start_ix, start_iy),
        "polygon_xy_m": pts,
    }

    os.makedirs("src/data/floor_plans", exist_ok=True)
    out_file = "../data/floor_plans/house_floorplan_pdf.pkl"

    with open(out_file, "wb") as f:
        pickle.dump(out, f)

    print(f"Saved floorplan pdf to: {out_file}")
    print(f"Map shape: {probability_map.shape}")
    print(f"Width x Height [m]: {width_m:.2f} x {height_m:.2f}")
    print(f"PDF sum: {probability_map.sum():.6f}")
    print(f"Start cell: {(start_ix, start_iy)}")

    # -------------------------------------------------
    # Quick visual check
    # -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(
        walkable,
        origin="lower",
        extent=[min_x, max_x, min_y, max_y],
        aspect="equal",
        cmap="Greys"
    )
    axes[0].plot(pts[:, 0], pts[:, 1], "r-", linewidth=2)
    axes[0].plot(start_xy[0], start_xy[1], "go", markersize=8)
    axes[0].set_title("Binary walkable map")

    axes[1].imshow(
        probability_map,
        origin="lower",
        extent=[min_x, max_x, min_y, max_y],
        aspect="equal"
    )
    axes[1].plot(pts[:, 0], pts[:, 1], "w-", linewidth=1.5)
    axes[1].plot(start_xy[0], start_xy[1], "go", markersize=8)
    axes[1].set_title("Smoothed normalized floorplan PDF")

    for ax in axes:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()