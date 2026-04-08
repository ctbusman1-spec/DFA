from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def main():
    # =========================================================
    # DIMENSIONS FROM YOUR MARKED FLOORPLAN (meters)
    # =========================================================

    # Main corridor
    corridor_w = 0.82
    corridor_h = 4.65

    # Lower left room
    left_room_w = 1.65
    left_room_h = 0.70
    left_room_bottom_y = 2.00   # chosen so it visually matches your drawing

    # Top room main block
    top_room_w = 1.90
    top_room_h = 1.92

    # Left top ledge / shoulder
    top_left_ledge_w = 1.60
    top_left_drop = 0.43

    # Middle left notch (the red one you drew)
    notch_left_w = 1.71
    notch_drop_1 = 0.88
    notch_step_right = 0.35
    notch_drop_2 = 1.35

    # Right top niche
    right_niche_w = 1.10
    right_niche_h = 0.82
    right_niche_drop = 0.30
    top_right_outer_h = 1.70

    # =========================================================
    # REFERENCE POSITIONING
    # =========================================================
    # Put corridor bottom-left at x=4.00, y=0.00
    cx0 = 4.00
    cy0 = 0.00

    cx1 = cx0 + corridor_w
    cy1 = cy0 + corridor_h

    # lower left room
    lx0 = cx0 - left_room_w
    lx1 = cx0
    ly0 = left_room_bottom_y
    ly1 = ly0 + left_room_h

    # top-left shoulder line
    shoulder_y = cy1
    shoulder_x0 = cx0 - notch_left_w
    shoulder_x1 = cx0

    # top room
    tx0 = cx0 - top_room_w + 0.02   # tiny shift so it visually aligns nicer
    tx1 = tx0 + top_room_w
    ty0 = shoulder_y + top_left_drop
    ty1 = ty0 + top_room_h

    # right niche
    rx0 = cx1
    rx1 = rx0 + right_niche_w
    ry0 = cy1
    ry1 = ry0 + right_niche_h

    # top outer right
    top_outer_right_y = cy1 + top_right_outer_h

    # notch geometry
    n1_x0 = shoulder_x0
    n1_x1 = shoulder_x0 + notch_left_w
    n1_y = shoulder_y

    n2_x = n1_x1
    n2_y = n1_y - notch_drop_1

    n3_x = n2_x + notch_step_right
    n3_y = n2_y

    n4_x = n3_x
    n4_y = n3_y - notch_drop_2

    # =========================================================
    # BUILD ORTHOGONAL OUTER CONTOUR CLOCKWISE
    # =========================================================
    pts = [
        # start at corridor bottom-left
        (cx0, cy0),
        (cx1, cy0),
        (cx1, ry0),            # up right side of corridor to niche bottom
        (rx1, ry0),            # niche out to the right
        (rx1, top_outer_right_y),
        (tx0, top_outer_right_y),
        (tx0, ty0),            # down left side of top room
        (shoulder_x0, shoulder_y),  # shoulder left
        (n1_x1, n1_y),         # long red horizontal to center
        (n2_x, n2_y),          # down
        (n3_x, n3_y),          # step right
        (n4_x, n4_y),          # down
        (lx1, ly1),            # to top-right of left room
        (lx0, ly1),            # left room top-left
        (lx0, ly0),            # left room bottom-left
        (cx0, ly0),            # back to corridor left edge
        (cx0, cy0),            # close
    ]

    fig, ax = plt.subplots(figsize=(7, 9))
    poly = Polygon(
        pts,
        closed=True,
        facecolor="#dfe8f5",
        edgecolor="black",
        linewidth=2.5,
        alpha=0.92,
    )
    ax.add_patch(poly)

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------
    ax.text((cx0 + cx1) / 2, 2.5, "hoofdgang", ha="center", va="center", fontsize=12)
    ax.text((lx0 + lx1) / 2, (ly0 + ly1) / 2, "kamer links", ha="center", va="center", fontsize=12)
    ax.text((tx0 + tx1) / 2, (ty0 + ty1) / 2, "topkamer", ha="center", va="center", fontsize=12)

    # Start point
    start_x = cx0 + corridor_w / 2
    start_y = 0.40
    ax.plot(start_x, start_y, "go", markersize=10, label="start")

    # ---------------------------------------------------------
    # Dimension annotations (simple)
    # ---------------------------------------------------------
    def dim_text(x, y, txt):
        ax.text(
            x, y, txt,
            ha="center", va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="gray", alpha=0.9),
        )

    dim_text((cx0 + cx1) / 2, 0.10, f"{corridor_w:.2f} m")
    dim_text(cx1 + 0.28, corridor_h / 2, f"{corridor_h:.2f} m")

    dim_text((lx0 + lx1) / 2, ly0 - 0.12, f"{left_room_w:.2f} m")
    dim_text(lx0 - 0.20, (ly0 + ly1) / 2, f"{left_room_h:.2f} m")

    dim_text((tx0 + tx1) / 2, ty1 + 0.10, f"{top_room_w:.2f} m")
    dim_text(tx0 - 0.22, (ty0 + ty1) / 2, f"{top_room_h:.2f} m")

    dim_text((shoulder_x0 + shoulder_x1) / 2, shoulder_y + 0.12, f"{notch_left_w:.2f} m")
    dim_text(n1_x1 - 0.12, (n1_y + n2_y) / 2, f"{notch_drop_1:.2f} m")
    dim_text((n2_x + n3_x) / 2, n2_y - 0.12, f"{notch_step_right:.2f} m")
    dim_text(n4_x + 0.18, (n3_y + n4_y) / 2, f"{notch_drop_2:.2f} m")

    dim_text((rx0 + rx1) / 2, ry1 + 0.08, f"{right_niche_w:.2f} m")
    dim_text(rx0 - 0.18, (ry0 + ry1) / 2, f"{right_niche_h:.2f} m")
    dim_text(rx1 + 0.20, (ry0 + top_outer_right_y) / 2, f"{top_right_outer_h:.2f} m")

    ax.set_aspect("equal")
    ax.set_xlim(2.0, 6.3)
    ax.set_ylim(0.0, 7.4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Polycam floorplan preview – corrected dimensions")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()