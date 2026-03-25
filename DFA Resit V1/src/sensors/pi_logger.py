from __future__ import annotations

import csv
import math
import os
import time
from datetime import datetime

from sense_hat import SenseHat


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


def ensure_output_dir(path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def main() -> None:
    sense = SenseHat()

    sample_rate_hz = 50
    dt_target = 1.0 / sample_rate_hz

    detector = StepDetector(threshold_g=1.18, min_interval_s=0.30)

    start_wall = time.time()
    prev_time = start_wall

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"src/data/experiments/pi_sensor_log_still.csv"
    ensure_output_dir(output_file)

    print(f"Logging to: {output_file}")
    print("Press Ctrl+C to stop.\n")

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

    n_rows = 0
    n_steps = 0
    last_status_print = start_wall

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        try:
            while True:
                now = time.time()
                timestamp = now - start_wall
                dt = now - prev_time
                prev_time = now

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

                writer.writerow(
                    {
                        "timestamp": round(timestamp, 6),
                        "dt": round(dt, 6),
                        "heading_rad": round(heading_rad, 6),
                        "accel_x": round(ax, 6),
                        "accel_y": round(ay, 6),
                        "accel_z": round(az, 6),
                        "accel_norm": round(accel_norm, 6),
                        "step_event": step_event,
                    }
                )
                f.flush()

                n_rows += 1

                if now - last_status_print > 1.0:
                    print(
                        f"t={timestamp:6.2f}s | rows={n_rows:5d} | "
                        f"steps={n_steps:3d} | accel_norm={accel_norm:.3f} | "
                        f"heading={heading_rad:.3f}"
                    )
                    last_status_print = now

                elapsed = time.time() - now
                sleep_time = max(0.0, dt_target - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopped logging.")
            print(f"Saved file: {output_file}")
            print(f"Total rows: {n_rows}")
            print(f"Detected steps: {n_steps}")


if __name__ == "__main__":
    main()