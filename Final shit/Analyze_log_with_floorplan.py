# analyze_log_with_floorplan.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

# =========================
# CHANGE THIS IN PYCHARM
# =========================
LOG_FILE = "imu_log_1.csv"   # <-- zet hier jouw lognaam (met .csv)
# bv: LOG_FILE = r"C:\Users\jij\Downloads\imu_log_1.csv"

# =========================
# Floorplan (NEW, fixed) - same geometry as Pi
# =========================
@dataclass
class Rect:
    x0: float
    y0: float
    w: float
    h: float

    def contains(self, x: float, y: float) -> bool:
        return (self.x0 <= x <= self.x0 + self.w) and (self.y0 <= y <= self.y0 + self.h)


class NewFloorplan:
    def __init__(self):
        self.left_block = Rect(0.0, 0.0, 2.0, 2.3)
        self.right_corridor = Rect(2.0, 0.0, 5.0, 0.8)

        vx0 = 2.0 + 5.0 - 0.8
        vy0 = 0.4  # <-- 40 cm lower fix
        vheight = 2.3
        self.vertical_corridor = Rect(vx0, vy0, 0.8, vheight)

        self.rects = [self.left_block, self.right_corridor, self.vertical_corridor]

    def is_walkable(self, x: float, y: float) -> bool:
        return any(r.contains(x, y) for r in self.rects)

    def bounds(self):
        xs = [r.x0 for r in self.rects] + [r.x0 + r.w for r in self.rects]
        ys = [r.y0 for r in self.rects] + [r.y0 + r.h for r in self.rects]
        return (min(xs), max(xs), min(ys), max(ys))


def summarize_states(df: pd.DataFrame):
    if "state" not in df.columns:
        print("[WARN] No 'state' column found.")
        return
    counts = df["state"].value_counts()
    total = len(df)
    print("\nSTATE DISTRIBUTION")
    for k, v in counts.items():
        print(f"  {k:>6}: {v:6d} ({100*v/total:5.1f}%)")

    changes = (df["state"].shift(1) != df["state"]).fillna(False)
    trans_idx = df.index[changes].tolist()
    print(f"\nTRANSITIONS: {max(0, len(trans_idx)-1)}")
    for idx in trans_idx[:8]:
        t = df.loc[idx, "t"] if "t" in df.columns else idx
        print(f"  t={t:.2f}s -> {df.loc[idx,'state']}")
    if len(trans_idx) > 8:
        print("  ...")


def summarize_steps(df: pd.DataFrame):
    if "step" not in df.columns:
        print("[WARN] No 'step' column found.")
        return
    step_rows = df[df["step"] == 1]
    nsteps = len(step_rows)
    duration = df["t"].iloc[-1] - df["t"].iloc[0] if ("t" in df.columns and len(df) > 1) else 0.0
    sps = nsteps / duration if duration > 0 else 0.0
    print("\nSTEP SUMMARY")
    print(f"  steps: {nsteps}")
    print(f"  duration: {duration:.2f}s")
    print(f"  steps/sec: {sps:.2f}")

    if nsteps >= 3 and "t" in df.columns:
        intervals = np.diff(step_rows["t"].values)
        print(f"  mean interval: {intervals.mean():.3f}s")
        print(f"  std interval : {intervals.std():.3f}s")
        if intervals.mean() > 0:
            print(f"  CV%         : {100*intervals.std()/intervals.mean():.1f}%")


def summarize_floorplan(df: pd.DataFrame, fp: NewFloorplan):
    if "pos_x_m" not in df.columns or "pos_y_m" not in df.columns:
        print("[WARN] No position columns found.")
        return

    xs = df["pos_x_m"].values
    ys = df["pos_y_m"].values
    walk = np.array([fp.is_walkable(x, y) for x, y in zip(xs, ys)], dtype=bool)

    outside = (~walk).sum()
    print("\nFLOORPLAN CONSISTENCY")
    print(f"  outside-walkable samples: {outside} / {len(df)} ({100*outside/len(df):.1f}%)")

    transitions_out = np.where((walk[:-1] == True) & (walk[1:] == False))[0]
    if len(transitions_out) > 0 and "t" in df.columns:
        print(f"  boundary crossings into invalid area: {len(transitions_out)}")
        for i in transitions_out[:5]:
            print(f"    t≈{df['t'].iloc[i]:.2f}s at pos ({xs[i]:.2f},{ys[i]:.2f})")
        if len(transitions_out) > 5:
            print("    ...")


def plot_all(df: pd.DataFrame, fp: NewFloorplan, out_png: str):
    t = df["t"].values if "t" in df.columns else np.arange(len(df))

    fig = plt.figure(figsize=(14, 10))

    # 1) accel magnitude
    ax1 = plt.subplot(3, 2, 1)
    if "acc_mag_g" in df.columns:
        ax1.plot(t, df["acc_mag_g"].values)
        ax1.set_title("Acceleration magnitude (g)")
        ax1.set_xlabel("time (s)")
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No acc_mag_g", ha="center", va="center")

    # 2) gyro z
    ax2 = plt.subplot(3, 2, 2)
    if "gyro_z_rads" in df.columns:
        ax2.plot(t, df["gyro_z_rads"].values)
        ax2.set_title("Gyro z (rad/s)")
        ax2.set_xlabel("time (s)")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No gyro_z_rads", ha="center", va="center")

    # 3) state timeline
    ax3 = plt.subplot(3, 2, 3)
    if "state" in df.columns:
        states = df["state"].astype(str).values
        uniq = list(dict.fromkeys(states))
        mapping = {s: i for i, s in enumerate(uniq)}
        y = np.array([mapping[s] for s in states])
        ax3.plot(t, y, linewidth=2)
        ax3.set_yticks(list(mapping.values()))
        ax3.set_yticklabels(list(mapping.keys()))
        ax3.set_title("State (STILL/MOVE)")
        ax3.set_xlabel("time (s)")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No state column", ha="center", va="center")

    # 4) steps on acc_mag
    ax4 = plt.subplot(3, 2, 4)
    if "acc_mag_g" in df.columns and "step" in df.columns:
        ax4.plot(t, df["acc_mag_g"].values, label="acc_mag_g")
        step_idx = df.index[df["step"] == 1].to_numpy()
        if len(step_idx) > 0:
            ax4.scatter(t[step_idx], df["acc_mag_g"].values[step_idx], s=30, label="step=1")
        ax4.set_title("Steps on acc magnitude")
        ax4.set_xlabel("time (s)")
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, "Need acc_mag_g + step", ha="center", va="center")

    # 5) trajectory over floorplan
    ax5 = plt.subplot(3, 2, 5)
    if "pos_x_m" in df.columns and "pos_y_m" in df.columns:
        for r in fp.rects:
            ax5.add_patch(plt.Rectangle((r.x0, r.y0), r.w, r.h, fill=False, linewidth=2))
        x = df["pos_x_m"].values
        y = df["pos_y_m"].values
        ax5.plot(x, y, linewidth=1.5)
        ax5.scatter([x[0]], [y[0]], s=60, marker="o", label="start")
        ax5.scatter([x[-1]], [y[-1]], s=60, marker="x", label="end")

        walk = np.array([fp.is_walkable(xx, yy) for xx, yy in zip(x, y)], dtype=bool)
        if (~walk).any():
            ax5.scatter(x[~walk], y[~walk], s=10, label="outside", alpha=0.6)

        xmin, xmax, ymin, ymax = fp.bounds()
        ax5.set_xlim(xmin - 0.5, xmax + 0.5)
        ax5.set_ylim(ymin - 0.5, ymax + 0.5)
        ax5.set_aspect("equal", adjustable="box")
        ax5.set_title("Dead-reckoning trajectory on NEW floorplan")
        ax5.set_xlabel("x (m)")
        ax5.set_ylabel("y (m)")
        ax5.grid(True, alpha=0.3)
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, "No pos_x_m/pos_y_m", ha="center", va="center")

    # 6) heading
    ax6 = plt.subplot(3, 2, 6)
    if "heading_rad" in df.columns:
        ax6.plot(t, df["heading_rad"].values)
        ax6.set_title("Heading (rad)")
        ax6.set_xlabel("time (s)")
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, "No heading_rad", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"[OK] Saved plot: {out_png}")
    plt.show()


def main():
    if not os.path.exists(LOG_FILE):
        raise FileNotFoundError(
            f"LOG_FILE not found: {LOG_FILE}\n"
            f"Tip: set LOG_FILE to an absolute path or put the CSV next to this script."
        )

    df = pd.read_csv(LOG_FILE)

    # Ensure numeric columns are numeric
    for c in ["t", "dt", "acc_mag_g", "gyro_z_rads", "heading_rad", "pos_x_m", "pos_y_m", "step"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"[INFO] Loaded {len(df)} rows from {LOG_FILE}")
    if "dt" in df.columns and df["dt"].notna().any():
        hz = 1.0 / df["dt"].mean()
        print(f"[INFO] Estimated Hz: {hz:.1f}")

    fp = NewFloorplan()

    summarize_states(df)
    summarize_steps(df)
    summarize_floorplan(df, fp)

    out_png = os.path.splitext(os.path.basename(LOG_FILE))[0] + "_analysis.png"
    plot_all(df, fp, out_png)


if __name__ == "__main__":
    main()
