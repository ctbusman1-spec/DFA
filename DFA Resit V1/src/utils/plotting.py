import matplotlib.pyplot as plt


def save_run_plot(run_df, floor_map, occupancy, png_path, title="trajectory + heat map"):
    fig, ax = plt.subplots(figsize=(8, 6))

    x0 = floor_map.origin_x_m
    x1 = floor_map.origin_x_m + floor_map.width_m
    y0 = floor_map.origin_y_m
    y1 = floor_map.origin_y_m + floor_map.height_m

    if occupancy is not None:
        ax.imshow(
            occupancy,
            origin="lower",
            extent=[x0, x1, y0, y1],
            aspect="equal",
            alpha=0.65,
        )

    if run_df is not None and not run_df.empty and {"x_m", "y_m"}.issubset(run_df.columns):
        ax.plot(run_df["x_m"], run_df["y_m"], marker="o", markersize=3, linewidth=1.5)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close(fig)


def save_comparison_plot(summary_df, png_path, title="model comparison"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    df = summary_df.copy()

    axes[0].bar(df["model"], df["path_length_m"])
    axes[0].set_title("Path length [m]")
    axes[0].set_ylabel("meters")

    axes[1].bar(df["model"], df["inside_map_ratio"])
    axes[1].set_title("Inside-map ratio")
    axes[1].set_ylabel("ratio")
    axes[1].set_ylim(0, 1.05)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close(fig)