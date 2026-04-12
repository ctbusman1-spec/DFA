from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def _normalize_heatmap(arr):
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    max_val = np.nanmax(arr)
    if not np.isfinite(max_val) or max_val <= 0:
        return arr
    return arr / max_val


def save_run_plot(run_df, floor_map, occupancy, png_path, title="trajectory over floorplan PDF"):
    fig, ax = plt.subplots(figsize=(8, 10))

    x0, x1, y0, y1 = floor_map.extent_xy

    ax.imshow(
        floor_map.probability_map,
        origin="lower",
        extent=[x0, x1, y0, y1],
        aspect="equal",
        alpha=0.95,
    )

    if occupancy is not None:
        occ = _normalize_heatmap(occupancy)
        ax.imshow(
            occ,
            origin="lower",
            extent=[x0, x1, y0, y1],
            aspect="equal",
            alpha=0.35,
        )

    if run_df is not None and not run_df.empty and {"x_m", "y_m"}.issubset(run_df.columns):
        ax.plot(run_df["x_m"], run_df["y_m"], marker="o", markersize=3, linewidth=1.6, label="Estimated trajectory")
        ax.scatter(run_df["x_m"].iloc[0], run_df["y_m"].iloc[0], s=70, marker="o", label="Start")
        ax.scatter(run_df["x_m"].iloc[-1], run_df["y_m"].iloc[-1], s=90, marker="x", label="End")

        if "heading_rad" in run_df.columns and len(run_df) > 2:
            step = max(1, len(run_df) // 20)
            idx = np.arange(0, len(run_df), step)
            arrow_len = 0.18
            u = arrow_len * np.cos(run_df["heading_rad"].iloc[idx])
            v = arrow_len * np.sin(run_df["heading_rad"].iloc[idx])
            ax.quiver(
                run_df["x_m"].iloc[idx],
                run_df["y_m"].iloc[idx],
                u,
                v,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.004,
            )

    if hasattr(floor_map, "polygon_xy_m") and floor_map.polygon_xy_m is not None:
        pts = floor_map.polygon_xy_m
        ax.plot(pts[:, 0], pts[:, 1], linestyle="--", linewidth=1.0, alpha=0.8, label="Walkable contour")

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=220)
    plt.close(fig)


def save_comparison_plot(summary_df, png_path, title="model comparison"):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    df = summary_df.copy()

    axes[0, 0].bar(df["model"], df["path_length_m"])
    axes[0, 0].set_title("Path length [m]")
    axes[0, 0].set_ylabel("meters")

    axes[0, 1].bar(df["model"], df["walkable_ratio"])
    axes[0, 1].set_title("Walkable ratio")
    axes[0, 1].set_ylabel("ratio")
    axes[0, 1].set_ylim(0, 1.05)

    axes[1, 0].bar(df["model"], df["runtime_ms_per_step"])
    axes[1, 0].set_title("Runtime per step [ms]")
    axes[1, 0].set_ylabel("ms")

    axes[1, 1].bar(df["model"], df["mean_neff"])
    axes[1, 1].set_title("Mean neff")
    axes[1, 1].set_ylabel("effective count")

    fig.suptitle(title)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=220)
    plt.close(fig)
