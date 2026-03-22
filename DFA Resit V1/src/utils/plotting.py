from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_run_plot(run_df: pd.DataFrame, floor_map, occupancy: np.ndarray, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.imshow(
        floor_map.pdf,
        origin="lower",
        extent=floor_map.extent(),
        alpha=0.45,
        aspect="auto",
    )
    if occupancy is not None and occupancy.sum() > 0:
        occ = occupancy / occupancy.max()
        ax.imshow(
            occ,
            origin="lower",
            extent=floor_map.extent(),
            alpha=0.45,
            aspect="auto",
        )
    ax.plot(run_df["x_m"], run_df["y_m"], marker="o", markersize=2, linewidth=1.5)
    ax.set_xlim(0, floor_map.width_m)
    ax.set_ylim(0, floor_map.height_m)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_comparison_plot(summary_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(summary_df))
    ax.bar(x - 0.18, summary_df["inside_map_ratio"], width=0.36, label="Inside-map ratio")
    ax.bar(x + 0.18, summary_df["path_length_m"], width=0.36, label="Path length [m]")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["model"])
    ax.legend()
    ax.set_title("Model comparison")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
