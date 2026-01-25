import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

CELL_CM = 10.0  # 10 cm per cell


def m_to_cell(m: float) -> int:
    return int(round((m * 100.0) / CELL_CM))


def add_rect(grid: np.ndarray, x0: int, y0: int, w: int, h: int, val: float):
    """Filled rectangle, grid[y, x]."""
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
    pad_m: float = 0.5,               # 50cm padding around everything
    mirror_vertically: bool = True,   # "zoals foto 3"
    right_corridor_drop_m: float = 0.40,  # <-- JOUW FIX: 40 cm lager starten
):
    """
    Walkable area = union of filled rectangles (no holes).
    Geometry based on your measured distances:

    - Left block width: 2.0 m
    - Left block height: 2.3 m
    - Corridor width everywhere: 0.8 m
    - Long right corridor length: 5.0 m (from x=2.0 to x=7.0)
    - Total width: 7.0 m (2m + 5m)
    - Vertical stem extends 1.5m below left block (total height 3.8m)

    Start position:
    - 1.0 m from left
    - 0.25 m from top
    """

    # ======================
    # Dimensions (meters)
    # ======================
    W_TOTAL = 7.0                 # 2m + 5m
    CORR_W = 0.8                  # 0.8m everywhere
    H_LEFT = 2.3                  # left block height
    H_EXTRA = 1.5                 # extra down for the stem
    H_TOTAL = H_LEFT + H_EXTRA    # 3.8m

    X_JUNCTION = 2.0              # junction at 2m from left
    RIGHT_LEN = 5.0               # right corridor length

    # Bottom step inside left block (from your earlier version)
    X_STEP = 1.4
    X_REMAIN = 0.6                # 1.4 + 0.6 = 2.0

    # Start position (meters)
    START_X = 1.0
    START_Y = 0.25

    # ======================
    # Canvas
    # ======================
    pad = m_to_cell(pad_m)
    W = m_to_cell(W_TOTAL + 2 * pad_m)
    H = m_to_cell(H_TOTAL + 2 * pad_m)
    floor = np.zeros((H, W), dtype=float)

    def to_grid(x_m: float, y_m: float):
        x = pad + m_to_cell(x_m)
        y = pad + m_to_cell(y_m)
        return x, y

    # ======================
    # 1) Left block (stays EXACTLY the same)
    # ======================
    x0, y0 = to_grid(0.0, 0.0)
    add_rect(
        floor,
        x0=x0,
        y0=y0,
        w=m_to_cell(X_JUNCTION),
        h=m_to_cell(H_LEFT),
        val=draw_value
    )

    # ======================
    # 2) Top “cap” corridor on the left (optional but helps match your blob-less top)
    #    This is just the 0.8m strip across the left block at the very top.
    # ======================
    x0, y0 = to_grid(0.0, 0.0)
    add_rect(
        floor,
        x0=x0,
        y0=y0,
        w=m_to_cell(X_JUNCTION),
        h=m_to_cell(CORR_W),
        val=draw_value
    )

    # ======================
    # 3) Long right corridor (5m x 0.8m) shifted DOWN by 0.40m
    #    This is your requested fix.
    # ======================
    x0, y0 = to_grid(X_JUNCTION, right_corridor_drop_m)
    add_rect(
        floor,
        x0=x0,
        y0=y0,
        w=m_to_cell(RIGHT_LEN),
        h=m_to_cell(CORR_W),
        val=draw_value
    )

    # ======================
    # 4) Vertical stem at the junction, also starts at that same dropped height
    #    (otherwise the connection looks wrong)
    # Stem spans x in [X_JUNCTION - 0.8, X_JUNCTION]
    # ======================
    stem_height = max(0.0, H_TOTAL - right_corridor_drop_m)
    x0, y0 = to_grid(X_JUNCTION - CORR_W, right_corridor_drop_m)
    add_rect(
        floor,
        x0=x0,
        y0=y0,
        w=m_to_cell(CORR_W),
        h=m_to_cell(stem_height),
        val=draw_value
    )

    # ======================
    # 5) Bottom step inside left block (keeps shape like your outline)
    # ======================
    # A) left bottom strip (1.4m wide, 0.8m thick) at y = H_LEFT - 0.8
    x0, y0 = to_grid(0.0, H_LEFT - CORR_W)
    add_rect(
        floor,
        x0=x0,
        y0=y0,
        w=m_to_cell(X_STEP),
        h=m_to_cell(CORR_W),
        val=draw_value
    )

    # B) connect strip from x=1.4 to x=2.0 (0.6m) same y
    x0, y0 = to_grid(X_STEP, H_LEFT - CORR_W)
    add_rect(
        floor,
        x0=x0,
        y0=y0,
        w=m_to_cell(X_REMAIN),
        h=m_to_cell(CORR_W),
        val=draw_value
    )

    # ======================
    # Smooth -> normalize to a probability map
    # ======================
    sm = gaussian_filter(floor, sigma=sigma_cells)
    if sm.max() > 0:
        sm = (sm / sm.max()) * max_prob_cap

    prior = sm.copy()
    s = prior.sum()
    if s > 0:
        prior /= s

    # ======================
    # Start
    # ======================
    sx, sy = to_grid(START_X, START_Y)

    # ======================
    # Mirror vertically (like photo 3)
    # ======================
    if mirror_vertically:
        prior = prior[::-1, :]
        sy = (H - 1) - sy

    meta = {
        "cell_cm": CELL_CM,
        "W": W,
        "H": H,
        "corridor_width_m": CORR_W,
        "sigma_cells": sigma_cells,
        "mirrored_vertically": mirror_vertically,
        "right_corridor_drop_m": right_corridor_drop_m
    }
    return prior, (sx, sy), meta


# ======================
# Quick test / plot
# ======================
if __name__ == "__main__":
    map_pdf, (xk, yk), meta = build_floorplan_prior_filled(
        sigma_cells=1.6,
        mirror_vertically=True,
        right_corridor_drop_m=0.40  # <-- 40cm fix
    )

    print(f"[INFO] Floorplan: {meta['W']}x{meta['H']} | start=({xk},{yk}) ")

    plt.figure(figsize=(9, 4))
    plt.imshow(map_pdf, origin="lower")
    plt.scatter([xk], [yk], s=60)
    plt.title("Floorplan prior + start")
    plt.tight_layout()
    plt.show()
