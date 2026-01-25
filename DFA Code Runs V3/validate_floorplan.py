#!/usr/bin/env python3
"""
Floorplan Validation - Complete Version
Uses inline floorplan definition (no external module dependency)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

print("\n" + "="*80)
print("FLOORPLAN VALIDATION")
print("="*80)

# Constants
CELL_CM = 10.0  # 10 cm per cell
CSV_FILE = 'Real-data-set-sensor_log_4.csv'

def m_to_cell(m: float) -> int:
    """Convert meters to cell count"""
    return int(round((m * 100.0) / CELL_CM))

def add_rect(grid: np.ndarray, x0: int, y0: int, w: int, h: int, val: float):
    """Draw a rectangle on the grid"""
    H, W = grid.shape
    x1 = max(0, min(W, x0 + w))
    y1 = max(0, min(H, y0 + h))
    x0 = max(0, min(W, x0))
    y0 = max(0, min(H, y0))
    grid[y0:y1, x0:x1] = np.maximum(grid[y0:y1, x0:x1], val)

def build_floorplan_prior_filled(
    sigma_cells: float = 1.6,
    draw_value: float = 20.0,
    max_prob_cap: float = 0.5,
    pad_m: float = 0.5,
    mirror_vertically: bool = True,
    right_corridor_drop_m: float = 0.40,
):
    """Build floorplan prior from room geometry"""
    # Cedric's room and hallway parameters
    W_TOTAL = 7.0
    CORR_W = 0.8
    H_LEFT = 2.3
    H_EXTRA = 1.5
    H_TOTAL = H_LEFT + H_EXTRA

    X_JUNCTION = 2.0
    RIGHT_LEN = 5.0

    X_STEP = 1.4
    X_REMAIN = 0.6

    START_X = 1.0
    START_Y = 0.70

    pad = m_to_cell(pad_m)
    W = m_to_cell(W_TOTAL + 2 * pad_m)
    H = m_to_cell(H_TOTAL + 2 * pad_m)
    floor = np.zeros((H, W), dtype=float)

    def to_grid(x_m: float, y_m: float):
        x = pad + m_to_cell(x_m)
        y = pad + m_to_cell(y_m)
        return x, y

    # Left block
    x0, y0 = to_grid(0.0, 0.0)
    add_rect(floor, x0, y0, m_to_cell(X_JUNCTION), m_to_cell(H_LEFT), draw_value)

    # Top cap strip
    x0, y0 = to_grid(0.0, 0.0)
    add_rect(floor, x0, y0, m_to_cell(X_JUNCTION), m_to_cell(CORR_W), draw_value)

    # Right corridor
    x0, y0 = to_grid(X_JUNCTION, right_corridor_drop_m)
    add_rect(floor, x0, y0, m_to_cell(RIGHT_LEN), m_to_cell(CORR_W), draw_value)

    # Vertical stem at junction
    stem_height = max(0.0, H_TOTAL - right_corridor_drop_m)
    x0, y0 = to_grid(X_JUNCTION - CORR_W, right_corridor_drop_m)
    add_rect(floor, x0, y0, m_to_cell(CORR_W), m_to_cell(stem_height), draw_value)

    # Bottom step inside left block
    x0, y0 = to_grid(0.0, H_LEFT - CORR_W)
    add_rect(floor, x0, y0, m_to_cell(X_STEP), m_to_cell(CORR_W), draw_value)

    x0, y0 = to_grid(X_STEP, H_LEFT - CORR_W)
    add_rect(floor, x0, y0, m_to_cell(X_REMAIN), m_to_cell(CORR_W), draw_value)

    # Smooth --> normalize
    sm = gaussian_filter(floor, sigma=sigma_cells)
    if sm.max() > 0:
        sm = (sm / sm.max()) * max_prob_cap

    prior = sm.copy()
    s = prior.sum()
    if s > 0:
        prior /= s

    sx, sy = to_grid(START_X, START_Y)

    if mirror_vertically:
        prior = prior[::-1, :]
        sy = (H - 1) - sy

    meta = {
        "cell_cm": CELL_CM,
        "W": W,
        "H": H,
        "pad_m": pad_m,
        "pad_cells": pad,
        "mirror_vertically": mirror_vertically,
        "right_corridor_drop_m": right_corridor_drop_m,
        "start_m": (START_X, START_Y),
        "start_cell": (sx, sy),
    }
    return prior, (sx, sy), meta

def meters_to_cell_xy(x_m: float, y_m: float, meta: dict):
    """Convert meters to cell coordinates"""
    pad = meta["pad_cells"]
    H = meta["H"]

    cx = pad + m_to_cell(x_m)
    cy = pad + m_to_cell(y_m)

    if meta["mirror_vertically"]:
        cy = (H - 1) - cy

    return cx, cy

def main():
    # Load results CSV
    print(f"\nLoading {CSV_FILE}...")
    if not os.path.exists(CSV_FILE):
        print(f"ERROR: {CSV_FILE} not found!")
        return
    
    df = pd.read_csv(CSV_FILE)
    print(f"✓ Loaded {len(df)} rows")

    # Build floorplan
    print("\nBuilding floorplan prior...")
    prior, (sx, sy), meta = build_floorplan_prior_filled(
        sigma_cells=1.6,
        mirror_vertically=True,
        right_corridor_drop_m=0.40
    )
    print(f"✓ Floorplan: {meta['W']}×{meta['H']} cells ({meta['W']*CELL_CM/100:.1f}m × {meta['H']*CELL_CM/100:.1f}m)")
    print(f"✓ Start position: ({sx:.0f}, {sy:.0f}) cells")

    # Convert ground truth trajectory to cells
    print("\nConverting trajectory to cell coordinates...")
    if "pos_x_m" in df.columns and "pos_y_m" in df.columns:
        traj_cells = np.array([meters_to_cell_xy(x, y, meta) for x, y in zip(df["pos_x_m"], df["pos_y_m"])])
        xs = traj_cells[:, 0]
        ys = traj_cells[:, 1]
        print(f"✓ Trajectory: {len(xs)} points")
    else:
        print("ERROR: pos_x_m and pos_y_m columns not found!")
        return

    # Check walkability
    print("\n" + "="*80)
    print("WALKABILITY ANALYSIS")
    print("="*80)
    
    H, W = prior.shape
    valid = []
    for x, y in zip(xs, ys):
        if 0 <= x < W and 0 <= y < H and prior[int(y), int(x)] > 0:
            valid.append(True)
        else:
            valid.append(False)
    
    valid = np.array(valid)
    outside = (~valid).sum()
    walkability = (valid.sum() / len(valid)) * 100
    
    print(f"\nGround Truth:")
    print(f"  Walkability: {walkability:.1f}% ({valid.sum()}/{len(valid)} points inside)")
    print(f"  Outside:     {outside} points")

    # Create visualization
    print("\n" + "="*80)
    print("CREATING VISUALIZATION")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Floorplan + trajectory
    ax = axes[0]
    im = ax.imshow(prior, origin="lower", cmap='gray', alpha=0.8)
    ax.scatter([sx], [sy], s=100, color='green', marker='o', label='Start', zorder=5)
    ax.plot(xs, ys, linewidth=2, color='blue', label='Ground Truth', alpha=0.7)
    
    # Highlight outside points
    if outside > 0:
        bad = np.where(~valid)[0]
        ax.scatter(xs[bad], ys[bad], s=20, color='red', alpha=0.6, label='Outside', zorder=4)
    
    ax.set_xlabel('X (cells)')
    ax.set_ylabel('Y (cells)')
    ax.set_title(f'Floorplan + Ground Truth Trajectory\nWalkability: {walkability:.1f}%')
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.colorbar(im, ax=ax, label='Walkable Probability')

    # Plot 2: Walkability over time
    ax = axes[1]
    time = df['t'].values if 't' in df.columns else np.arange(len(valid))
    ax.plot(time, valid.astype(int) * 100, linewidth=1, color='blue', label='Walkable')
    ax.axhline(y=walkability, color='red', linestyle='--', linewidth=2, label=f'Average: {walkability:.1f}%')
    ax.fill_between(time, 0, valid.astype(int) * 100, alpha=0.2, color='blue')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Walkability (%)')
    ax.set_title('Walkability Over Time')
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = 'floorplan_validation.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {out_png}")

    # Save validation report
    report_file = 'floorplan_validation_report.txt'
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FLOORPLAN VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Floorplan Dimensions: {meta['W']}×{meta['H']} cells\n")
        f.write(f"Cell Size: {CELL_CM} cm\n")
        f.write(f"Floorplan Size: {meta['W']*CELL_CM/100:.2f}m × {meta['H']*CELL_CM/100:.2f}m\n")
        f.write(f"Start Position: ({sx:.0f}, {sy:.0f}) cells = {meta['start_m']} m\n\n")
        f.write("WALKABILITY RESULTS:\n")
        f.write(f"  Total Samples:    {len(valid)}\n")
        f.write(f"  Inside Walkable:  {valid.sum()}\n")
        f.write(f"  Outside Walkable: {outside}\n")
        f.write(f"  Walkability:      {walkability:.2f}%\n")
    
    print(f"✓ Saved: {report_file}")

    print("\n" + "="*80)
    print("VALIDATION COMPLETE ✓")
    print("="*80)
    print(f"\nWalkability: {walkability:.1f}%")
    print(f"Status: {'✓ GOOD' if walkability > 90 else '⚠ ACCEPTABLE' if walkability > 80 else '✗ POOR'}\n")

if __name__ == "__main__":
    main()