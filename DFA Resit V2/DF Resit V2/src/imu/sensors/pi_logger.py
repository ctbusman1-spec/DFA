from __future__ import annotations


import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import csv
import math
import time
from pathlib import Path

from utils.paths import PROJECT_ROOT

try:
    from sense_hat import SenseHat
except Exception:
    SenseHat = None


class StepDetector:
    def __init__(self, threshold_g: float = 1.18, min_interval_s: float = 0.30):
        self.threshold_g = threshold_g
        self.min_interval_s = min_interval_s
        self.last_step_time = -1e9
        self.prev_above = False

    def update(self, accel_norm: float, timestamp: float) -> int:
        above = accel_norm > self.threshold_g
        rising_edge = above and not self.prev_above
        enough_time = (timestamp - self.last_step_time) >= self.min_interval_s
        step_event = 1 if (rising_edge and enough_time) else 0
        if step_event:
            self.last_step_time = timestamp
        self.prev_above = above
        return step_event


def main():
    if SenseHat is None:
        raise RuntimeError("sense_hat is not available on this machine.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pi_sensor_log")
    parser.add_argument("--sample-rate-hz", type=int, default=50)
    parser.add_argument("--threshold-g", type=float, default=1.18)
    parser.add_argument("--min-step-interval-s", type=float, default=0.30)
    parser.add_argument("--stop-after-inactive-s", type=float, default=10.0)
    args = parser.parse_args()

    sense = SenseHat()
    dt_target = 1.0 / args.sample_rate_hz
    detector = StepDetector(threshold_g=args.threshold_g, min_interval_s=args.min_step_interval_s)

    output_file = PROJECT_ROOT / "data" / "experiments" / f"{args.name}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Logging to: {output_file}")
    print("Logger stops automatically after 10 seconds without a detected step.")
    print("Press Ctrl+C to stop earlier.\n")

    fieldnames = [
        "timestamp",
        "dt",
        "heading_rad",
        "accel_x",
        "accel_y",
        "accel_z",
        "accel_norm",
        "step_event",
    ]

    start_wall = time.time()
    prev_time = start_wall
    last_step_timestamp = start_wall
    n_rows = 0
    n_steps = 0
    last_status_print = start_wall

    with output_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        try:
            while True:
                loop_start = time.time()
                timestamp = loop_start - start_wall
                dt = loop_start - prev_time
                prev_time = loop_start

                acc = sense.get_accelerometer_raw()
                ori = sense.get_orientation_radians()

                ax = float(acc["x"])
                ay = float(acc["y"])
                az = float(acc["z"])
                heading_rad = float(ori["yaw"])

                accel_norm = math.sqrt(ax * ax + ay * ay + az * az)
                step_event = detector.update(accel_norm, timestamp)

                if step_event:
                    n_steps += 1
                    last_step_timestamp = loop_start

                writer.writerow({
                    "timestamp": round(timestamp, 6),
                    "dt": round(dt, 6),
                    "heading_rad": round(heading_rad, 6),
                    "accel_x": round(ax, 6),
                    "accel_y": round(ay, 6),
                    "accel_z": round(az, 6),
                    "accel_norm": round(accel_norm, 6),
                    "step_event": step_event,
                })
                f.flush()
                n_rows += 1

                if loop_start - last_status_print >= 1.0:
                    print(
                        f"t={timestamp:6.2f}s | rows={n_rows:5d} | steps={n_steps:3d} | "
                        f"accel_norm={accel_norm:.3f} | heading={heading_rad:.3f}"
                    )
                    last_status_print = loop_start

                if (loop_start - last_step_timestamp) >= args.stop_after_inactive_s:
                    print(f"\nStopped automatically after {args.stop_after_inactive_s:.1f}s without activity.")
                    break

                elapsed = time.time() - loop_start
                time.sleep(max(0.0, dt_target - elapsed))

        except KeyboardInterrupt:
            print("\nStopped by user.")

    print(f"Saved file: {output_file}")
    print(f"Total rows: {n_rows}")
    print(f"Detected steps: {n_steps}")


if __name__ == "__main__":
    main()
