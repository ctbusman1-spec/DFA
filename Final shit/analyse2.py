import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# =========================
# CHANGE THESE IN PYCHARM
# =========================
LOG_FILE = "imu_log_7.csv"

START_X_M_FROM_LEFT = 1.0   # meters from LEFT wall
START_Y_M_FROM_TOP  = 0.80  # meters from TOP (your new start)

CELL_CM = 10.0  # 10 cm per cell


# -------------------------
# Floorplan prior (filled rectangles) - same geometry, new START
# -------------------------
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
    start_x_m: float = 1.0,
    start_y_from_top_m: float = 0.25,   # IMPORTANT: from TOP
):
    """
    Walkable area = union of filled rectangles.
    Coordinates in meters:
    - x: left -> right
    - y: top -> down for specifying start_y_from_top_m (then converted)

    The prior grid is mirrored vertically at the end to match your reference figure.
    """

    # Geometry
    W_TOTAL = 7.0
    CORR_W = 0.8
    H_LEFT = 2.3
    H_EXTRA = 1.5
    H_TOTAL = H_LEFT + H_EXTRA

    X_JUNCTION = 2.0
    RIGHT_LEN = 5.0

    X_STEP = 1.4
    X_REMAIN = 0.6

    # Convert "start from top" to "y from bottom" in the model coordinates (before mirroring)
    start_y_from_bottom_m = H_TOTAL - start_y_from_top_m

    pad = m_to_cell(pad_m)
    W = m_to_cell(W_TOTAL + 2 * pad_m)
    H = m_to_cell(H_TOTAL + 2 * pad_m)
    floor = np.zeros((H, W), dtype=float)

    def to_grid(x_m: float, y_m_from_bottom: float):
        x = pad + m_to_cell(x_m)
        y = pad + m_to_cell(y_m_from_bottom)
        return x, y

    # 1) Left block
    x0, y0 = to_grid(0.0, 0.0)
    add_rect(floor, x0, y0, m_to_cell(X_JUNCTION), m_to_cell(H_LEFT), draw_value)

    # 2) Top cap strip
    x0, y0 = to_grid(0.0, H_TOTAL - CORR_W)
    add_rect(floor, x0, y0, m_to_cell(X_JUNCTION), m_to_cell(CORR_W), draw_value)

    # 3) Right corridor shifted DOWN by 0.40m (down = toward bottom)
    # In "from bottom" coordinates: shift means corridor starts LOWER => smaller y_from_bottom
    right_y_from_bottom = H_TOTAL - CORR_W - right_corridor_drop_m
    x0, y0 = to_grid(X_JUNCTION, right_y_from_bottom)
    add_rect(floor, x0, y0, m_to_cell(RIGHT_LEN), m_to_cell(CORR_W), draw_value)

    # 4) Vertical stem at junction
    stem_height = max(0.0, H_TOTAL - right_corridor_drop_m)
    stem_y_from_bottom = H_TOTAL - stem_height - right_corridor_drop_m  # align connection
    x0, y0 = to_grid(X_JUNCTION - CORR_W, stem_y_from_bottom)
    add_rect(floor, x0, y0, m_to_cell(CORR_W), m_to_cell(stem_height), draw_value)

    # 5) Bottom step inside left block (keep your shape)
    x0, y0 = to_grid(0.0, 0.0)
    add_rect(floor, x0, y0, m_to_cell(X_STEP), m_to_cell(CORR_W), draw_value)

    x0, y0 = to_grid(X_STEP, 0.0)
    add_rect(floor, x0, y0, m_to_cell(X_REMAIN), m_to_cell(CORR_W), draw_value)

    # Smooth -> normalize
    sm = gaussian_filter(floor, sigma=sigma_cells)
    if sm.max() > 0:
        sm = (sm / sm.max()) * max_prob_cap

    prior = sm.copy()
    s = prior.sum()
    if s > 0:
        prior /= s

    # Start cell (before mirroring)
    sx, sy = to_grid(start_x_m, start_y_from_bottom_m)

    # Mirror vertically to match your figure style
    if mirror_vertically:
        prior = prior[::-1, :]
        sy = (H - 1) - sy

    meta = {
        "cell_cm": CELL_CM,
        "W": W,
        "H": H,
        "pad_m": pad_m,
        "pad_cells": pad,
        "H_TOTAL": H_TOTAL,
        "mirror_vertically": mirror_vertically,
        "start_m": (start_x_m, start_y_from_top_m),
        "start_cell": (sx, sy),
    }
    return prior, (sx, sy), meta


# -------------------------
# Convert meters -> cells for trajectory overlay (aligned-to-start)
# -------------------------
def meters_to_cell_xy(x_m: float, y_m_from_top: float, meta: dict):
    """
    Convert (x from left, y from top) to cell coords on the mirrored prior.
    """
    pad = meta["pad_cells"]
    H = meta["H"]
    H_TOTAL = meta["H_TOTAL"]

    y_from_bottom = H_TOTAL - y_m_from_top

    cx = pad + m_to_cell(x_m)
    cy = pad + m_to_cell(y_from_bottom)

    if meta["mirror_vertically"]:
        cy = (H - 1) - cy

    return cx, cy


def main():
    if not os.path.exists(LOG_FILE):
        raise FileNotFoundError(f"LOG_FILE not found: {LOG_FILE}")

    df = pd.read_csv(LOG_FILE)
    print(f"[INFO] Loaded {len(df)} rows from {LOG_FILE}")

    # Hz estimate
    hz = None
    if "dt" in df.columns and (df["dt"] > 0).any():
        hz = 1.0 / df.loc[df["dt"] > 0, "dt"].mean()
        print(f"[INFO] Estimated Hz: {hz:.1f}")

    # Build floorplan prior with YOUR NEW START
    prior, (sx, sy), meta = build_floorplan_prior_filled(
        sigma_cells=1.6,
        mirror_vertically=True,
        right_corridor_drop_m=0.40,
        start_x_m=START_X_M_FROM_LEFT,
        start_y_from_top_m=START_Y_M_FROM_TOP
    )

    # ---------
    # Trajectory: take DR meters from CSV, rebase to start, then map to cells
    # ---------
    required = {"pos_x_m", "pos_y_m"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {required - set(df.columns)}")

    # Rebase (shape check): make first sample correspond to map start
    x0 = float(df["pos_x_m"].iloc[0])
    y0 = float(df["pos_y_m"].iloc[0])

    # Interpret df pos_y_m as "meters from top" OR "from some origin"?
    # Your logger uses arbitrary meters. We rebase and then treat as offsets in map frame:
    df["x_map_m"] = (df["pos_x_m"] - x0) + START_X_M_FROM_LEFT
    df["y_map_m_from_top"] = (df["pos_y_m"] - y0) + START_Y_M_FROM_TOP

    traj_cells = np.array([
        meters_to_cell_xy(x, y, meta)
        for x, y in zip(df["x_map_m"], df["y_map_m_from_top"])
    ])
    xs = traj_cells[:, 0]
    ys = traj_cells[:, 1]

    # Walkable check: prior>0
    H, W = prior.shape
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    inside = np.zeros(len(xs), dtype=bool)
    inside[valid] = prior[ys[valid].astype(int), xs[valid].astype(int)] > 0
    outside = (~inside).sum()
    print(f"[INFO] Outside-walkable (prior-based): {outside}/{len(xs)} ({100*outside/len(xs):.1f}%)")

    # Step/stride sanity
    if "step" in df.columns:
        steps = int(df["step"].sum())
        duration = float(df["t"].iloc[-1]) if "t" in df.columns else (len(df)/hz if hz else np.nan)
        print("\nSTEP SUMMARY")
        print(f"  steps: {steps}")
        if duration and duration > 0:
            print(f"  duration: {duration:.2f}s")
            print(f"  steps/sec: {steps/duration:.2f}")
    else:
        steps = None

    # Approx DR path length in meters (map-rebased)
    dx = np.diff(df["x_map_m"].values)
    dy = np.diff(df["y_map_m_from_top"].values)
    path_len = float(np.sum(np.sqrt(dx*dx + dy*dy)))
    print(f"\nPATH LENGTH (from DR samples): {path_len:.2f} m")

    # ---------
    # Plot
    # ---------
    plt.figure(figsize=(10, 5))
    plt.imshow(prior, origin="lower")
    plt.scatter([sx], [sy], s=60, label="start")
    plt.plot(xs, ys, linewidth=3, label="trajectory")

    # show outside points
    bad = np.where(~inside)[0]
    if len(bad) > 0:
        plt.scatter(xs[bad], ys[bad], s=10, label="outside", alpha=0.7)

    plt.title("Floorplan prior + trajectory (cell coords)")
    plt.legend()

    # IMPORTANT: keep map aspect (no flattening)
    plt.xlim(0, prior.shape[1]-1)
    plt.ylim(0, prior.shape[0]-1)
    plt.gca().set_aspect("equal", adjustable="box")

    out_png = os.path.splitext(os.path.basename(LOG_FILE))[0] + "_prior_overlay_v2.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    print(f"\n[OK] Saved: {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
