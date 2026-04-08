from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def main():
    fig, ax = plt.subplots(figsize=(8, 9))

    # Polygon punten met de klok mee, grof gebaseerd op jouw rode contour
    # x naar rechts, y naar boven, in meters
    pts = [
        (4.00, 0.00),  # onder links hoofdgang
        (4.85, 0.00),  # onder rechts hoofdgang
        (4.85, 4.60),  # boven rechts hoofdgang
        (4.65, 4.60),   # deur naar
        (4.65, 4.65),
        (5.05, 4.65),
        (5.05, 5.35),  # hoek van kast
        (5.65, 5.35),  # klein stukje naar muur bureau
        (5.65, 6.90),  # helemaal omhoog

        (3.95, 6.90),  # bovenrand topkamer naar links
        (3.95, 5.25),  # omlaag topkamer
        (2.15, 5.25),  # links topkamer inkeping
        (2.15, 4.65),  # klein stukje omlaag
        (3.85, 4.65),  # terug naar rechterkolom

        (3.85, 3.90),  # omlaag smalle inkeping
        (4.00, 3.90),  # klein naar links
        (4.00, 2.80),  # omlaag tot middendeel
        (2.35, 2.80),  # links middenkamer bovenrand
        (2.35, 2.00),  # omlaag middenkamer
        (4.00, 2.00),  # terug naar hoofdgang
    ]

    poly = Polygon(
        pts,
        closed=True,
        facecolor="#dfe8f5",
        edgecolor="black",
        linewidth=2.5,
        alpha=0.9
    )
    ax.add_patch(poly)

    # Startpunt
    start_x = 5.25
    start_y = 5.85
    ax.plot(start_x, start_y, "go", markersize=10, label="start")

    # Handige labels
    ax.text(4.45, 2.4, "hoofdgang", ha="center", va="center", fontsize=11)
    ax.text(3.2, 5.6, "topkamer", ha="center", va="center", fontsize=11)
    ax.text(3.0, 2.45, "kamer links", ha="center", va="center", fontsize=11)

    ax.set_aspect("equal")
    ax.set_xlim(1.8, 5.8)
    ax.set_ylim(0.0, 7.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Preview vloercontour op basis van rode schets")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()