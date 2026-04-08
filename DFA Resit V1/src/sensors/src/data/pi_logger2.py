from __future__ import annotations

import argparse
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


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def ensure_output_dir(path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None, help="Custom log file name without .csv")
    parser.add_argument("--sample-rate", type=int, default=50)
    parser.add_argument("--threshold-g", type=float, default=1.18)
    parser.add_argument("--min-step-interval", type=float, default=0.30)
    parser.add_argument("--idle-stop-seconds", type=float, default=10.0)
    parser.add_argument("--heading-activity-threshold", type=float, default=0.08)
    args = parser.parse_args()

    sense = SenseHat()

    dt_target = 1.0 / args.sample_rate
    detector = StepDetector(
        threshold_g=args.threshold_g,
        min_interval_s=args.min_step_interval,
    )

    start_wall = time.time()
    prev_time = start_wall
    last_activity_time = start_wall
    prev_heading = None

    if args.name:
        filename = f"{args.name}.csv"
    else:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pi_sensor_log_straight1.csv"

    output_file = f"src/data/experiments/{filename}"
    ensure_output_dir(output_file)

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

                heading_activity = False
                if prev_heading is not None:
                    d_heading = abs(wrap_angle(heading_rad - prev_heading))
                    heading_activity = d_heading >= args.heading_activity_threshold
                prev_heading = heading_rad

                if step_event:
                    n_steps += 1
                    last_activity_time = now
                elif heading_activity:
                    last_activity_time = now

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

                idle_time = now - last_activity_time
                if idle_time >= args.idle_stop_seconds:
                    break

                elapsed = time.time() - now
                time.sleep(max(0.0, dt_target - elapsed))

        except KeyboardInterrupt:
            pass

    print(f"Saved file: {output_file}")
    print(f"Total rows: {n_rows}")
    print(f"Detected steps: {n_steps}")


if __name__ == "__main__":
    main()
