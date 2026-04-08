from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def main():
    fig, ax = plt.subplots(figsize=(7, 9))

    # Aangepaste contour op basis van jouw rode correctie
    pts = [
        # onderkant hoofdgang
        (4.00, 0.00),
        (4.82, 0.00),

        # rechterzijde hoofdgang omhoog
        (4.82, 4.72),

        # rechter nis / bovenkamer rechterstuk
        (5.95, 4.72),
        (5.95, 5.18),
        (4.72, 5.18),

        # kleine inkeping links van rechterstuk
        (4.72, 4.78),
        (4.58, 4.78),
        (4.58, 4.05),

        # midden-inkeping
        (4.05, 4.05),
        (4.05, 3.82),
        (4.20, 3.82),
        (4.20, 2.72),

        # kamer links
        (2.52, 2.72),
        (2.52, 2.02),
        (4.00, 2.02),

        # terug omhoog langs hoofdgang links
        (4.00, 4.62),

        # bovenzijde linker arm
        (2.42, 4.62),

        # linksboven schuine/gebogen indruk benaderen met 2 segmenten
        (2.42, 4.82),
        (2.18, 5.18),
        (2.18, 6.22),

        # topkamer bovenzijde
        (4.02, 6.22),
        (4.02, 5.18),

        # terug naar de lange horizontale arm
        (3.72, 5.18),
        (3.72, 4.62),

        # sluiten
        (4.00, 4.62),
        (4.00, 0.00),
    ]

    poly = Polygon(
        pts,
        closed=True,
        facecolor="#dfe8f5",
        edgecolor="black",
        linewidth=2.5,
        alpha=0.92,
    )
    ax.add_patch(poly)

    # labels
    ax.text(4.40, 2.35, "hoofdgang", ha="center", va="center", fontsize=12)
    ax.text(3.05, 2.35, "kamer links", ha="center", va="center", fontsize=12)
    ax.text(3.05, 5.75, "topkamer", ha="center", va="center", fontsize=12)

    # start
    start_x = 4.40
    start_y = 0.40
    ax.plot(start_x, start_y, "go", markersize=10, label="start")

    ax.set_aspect("equal")
    ax.set_xlim(2.0, 6.3)
    ax.set_ylim(0.0, 7.0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Polycam floorplan preview – contour adjusted")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
