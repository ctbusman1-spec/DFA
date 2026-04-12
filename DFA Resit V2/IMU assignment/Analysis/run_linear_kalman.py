from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Project paths
ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis.linear_kalman_filter import LinearKalmanFilter



# Paths
LOG_DIR = PROJECT_ROOT / "data" / "logs"
OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments" / "outputs" / "kalman_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# File selection
PATTERNS = {
    "still": "sensor_log_still*.csv",
    "walk": "sensor_log_walk*.csv",
    "shortTurn": "sensor_log_short_turn*.csv",
    "180Turn": "sensor_log_turn180*.csv",
}


def pick_first_matching_file(folder: Path, pattern: str) -> Path | None:
    files = sorted(folder.glob(pattern))
    return files[0] if files else None


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_log(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = ["timestamp", "dt", "gyro_x_rads", "gyro_y_rads", "gyro_z_rads"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "dt"])
    for col in ["gyro_x_rads", "gyro_y_rads", "gyro_z_rads"]:
        df[col] = df[col].fillna(0.0)

    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# Kalman filter application
# ---------------------------------------------------------------------
def apply_linear_kf_to_gyro(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    dt_med = float(np.nanmedian(df["dt"].to_numpy()))
    if not math.isfinite(dt_med) or dt_med <= 0:
        dt_med = 0.05

    kf = LinearKalmanFilter(
        dt=dt_med,
        process_noise_accel=0.05,
        process_noise_drift=0.01,
        measurement_noise_accel=0.08,
        measurement_noise_velocity=0.05,
        measurement_noise_zupt=0.01,
    )
    kf.reset()
    kf.initialize_state(position=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0])

    gx_f, gy_f, gz_f = [], [], []

    for _, row in df.iterrows():
        dt = float(row["dt"])
        if math.isfinite(dt) and dt > 0:
            kf.dt = dt
            kf.F = np.array([
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ], dtype=np.float32)

        kf.predict()

        z = np.array([
            float(row["gyro_x_rads"]),
            float(row["gyro_y_rads"]),
            float(row["gyro_z_rads"]),
        ], dtype=np.float32)

        if np.all(np.isfinite(z)):
            kf.update_velocity(z)

        _, vel = kf.get_state()
        gx_f.append(float(vel[0]))
        gy_f.append(float(vel[1]))
        gz_f.append(float(vel[2]))

    out = df.copy()
    out["gyro_x_kf"] = gx_f
    out["gyro_y_kf"] = gy_f
    out["gyro_z_kf"] = gz_f

    out["yaw_change_raw"] = np.cumsum(out["gyro_z_rads"].to_numpy() * out["dt"].to_numpy())
    out["yaw_change_kf"] = np.cumsum(out["gyro_z_kf"].to_numpy() * out["dt"].to_numpy())

    return out


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def save_signal_plot(df: pd.DataFrame, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["timestamp"], df["gyro_z_rads"], label="Raw gyro z")
    ax.plot(df["timestamp"], df["gyro_z_kf"], label="Kalman-filtered gyro z")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angular rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_yaw_plot(df: pd.DataFrame, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["timestamp"], df["yaw_change_raw"], label="Integrated raw gyro z")
    ax.plot(df["timestamp"], df["yaw_change_kf"], label="Integrated Kalman-filtered gyro z")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Accumulated yaw change")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------
def summarize_run(df: pd.DataFrame, group_name: str, run_name: str) -> dict:
    if df.empty:
        return {
            "group": group_name,
            "run_name": run_name,
            "duration_s": 0.0,
            "raw_gyro_z_std": np.nan,
            "kf_gyro_z_std": np.nan,
            "raw_final_yaw_change": np.nan,
            "kf_final_yaw_change": np.nan,
        }

    return {
        "group": group_name,
        "run_name": run_name,
        "duration_s": float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]),
        "raw_gyro_z_std": float(df["gyro_z_rads"].std()),
        "kf_gyro_z_std": float(df["gyro_z_kf"].std()),
        "raw_final_yaw_change": float(df["yaw_change_raw"].iloc[-1]),
        "kf_final_yaw_change": float(df["yaw_change_kf"].iloc[-1]),
    }


def build_markdown(summary_df: pd.DataFrame) -> str:
    lines = []
    lines.append("# Linear Kalman Filter Analysis")
    lines.append("")
    lines.append("This analysis applies the linear Kalman filter to the first raw sensor log in each experiment category. The Kalman filter is used here as a sensor-level smoothing method on gyroscope signals, not as the main map-based localization model.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("```text")
    lines.append(summary_df.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- **still** should show low gyroscope variance and low integrated yaw drift.")
    lines.append("- **walk** should remain relatively stable with smoother filtered gyro signals.")
    lines.append("- **shortTurn** should preserve the main turn while reducing local noise.")
    lines.append("- **180Turn** can be used as a challenge case: the Kalman filter smooths the signal, but does not solve the full localization problem on its own.")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    summaries = []

    for group_name, pattern in PATTERNS.items():
        csv_path = pick_first_matching_file(LOG_DIR, pattern)
        if csv_path is None:
            print(f"Skipping {group_name}: no matching raw log found in {LOG_DIR}")
            continue

        print(f"Processing {group_name}: {csv_path.name}")

        df = load_log(csv_path)
        df_f = apply_linear_kf_to_gyro(df)

        stem = f"{group_name}_{csv_path.stem}"

        filtered_csv = OUTPUT_DIR / f"{stem}_kalman_filtered.csv"
        gyro_plot = OUTPUT_DIR / f"{stem}_gyro_comparison.png"
        yaw_plot = OUTPUT_DIR / f"{stem}_yaw_integration.png"

        df_f.to_csv(filtered_csv, index=False)
        save_signal_plot(df_f, gyro_plot, title=f"{group_name}: raw vs Kalman-filtered gyro z")
        save_yaw_plot(df_f, yaw_plot, title=f"{group_name}: integrated raw vs Kalman-filtered gyro z")

        summaries.append(summarize_run(df_f, group_name, csv_path.stem))

    summary_df = pd.DataFrame(summaries)
    summary_csv = OUTPUT_DIR / "kalman_first_runs_summary.csv"
    summary_md = OUTPUT_DIR / "kalman_first_runs_analysis.md"

    summary_df.to_csv(summary_csv, index=False)
    summary_md.write_text(build_markdown(summary_df), encoding="utf-8")

    print(f"\nWrote: {summary_csv}")
    print(f"Wrote: {summary_md}")
    print(f"All plots and filtered CSVs are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()