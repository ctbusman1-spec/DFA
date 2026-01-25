import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

LOG_FILE = "imu_log_78.csv"

# floorplan code
CELL_CM = 10.0  # 10 cm per cell

# convert to meters
def m_to_cell(m: float) -> int:
    return int(round((m * 100.0) / CELL_CM))

def add_rect(grid: np.ndarray, x0: int, y0: int, w: int, h: int, val: float):
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
    START_Y = 0.7

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



# map meters -> cells
def meters_to_cell_xy(x_m: float, y_m: float, meta: dict):
    pad = meta["pad_cells"]
    H = meta["H"]

    cx = pad + m_to_cell(x_m)
    cy = pad + m_to_cell(y_m)

    if meta["mirror_vertically"]:
        cy = (H - 1) - cy

    return cx, cy


def main():
    if not os.path.exists(LOG_FILE):
        raise FileNotFoundError(f"LOG_FILE not found: {LOG_FILE}")

    df = pd.read_csv(LOG_FILE)
    print(f"[INFO] Loaded {len(df)} rows from {LOG_FILE}")

    if "dt" in df.columns and (df["dt"] > 0).any():
        hz = 1.0 / df.loc[df["dt"] > 0, "dt"].mean()
        print(f"[INFO] Estimated Hz: {hz:.1f}")

    # build the floorplan
    prior, (sx, sy), meta = build_floorplan_prior_filled(
        sigma_cells=1.6,
        mirror_vertically=True,
        right_corridor_drop_m=0.40
    )

    # Convert trajectory meters -> cells
    if "pos_x_m" in df.columns and "pos_y_m" in df.columns:
        traj_cells = np.array([meters_to_cell_xy(x, y, meta) for x, y in zip(df["pos_x_m"], df["pos_y_m"])])
        xs = traj_cells[:, 0]
        ys = traj_cells[:, 1]
    else:
        xs, ys = None, None

    # walkable check in CELL space
    outside = None
    if xs is not None:
        valid = []
        H, W = prior.shape
        for x, y in zip(xs, ys):
            if 0 <= x < W and 0 <= y < H and prior[int(y), int(x)] > 0:
                valid.append(True)
            else:
                valid.append(False)
        valid = np.array(valid)
        outside = (~valid).sum()
        print(f"[INFO] Outside-walkable (prior-based): {outside}/{len(valid)} ({100*outside/len(valid):.1f}%)")


    # Plot 1: Prior + trajectory overlay
    plt.figure(figsize=(10, 4))
    plt.imshow(prior, origin="lower")  # same style as your test
    plt.scatter([sx], [sy], s=60, label="start")

    if xs is not None:
        plt.plot(xs, ys, linewidth=3, label="trajectory")
        if outside is not None and outside > 0:
            bad = np.where(~valid)[0]
            plt.scatter(xs[bad], ys[bad], s=8, label="outside", alpha=0.7)

    #plot results
    plt.title("Floorplan + trajectory (cell coords)")
    plt.legend()
    plt.tight_layout()
    out_png = os.path.splitext(os.path.basename(LOG_FILE))[0] + "_overlay.png"
    plt.savefig(out_png, dpi=160)
    print(f"Saved: {out_png}")
    plt.show()



if __name__ == "__main__":
    main()
