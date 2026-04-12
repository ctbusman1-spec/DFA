from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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


def stop_requested_from_joystick(sense: SenseHat) -> bool:
    """
    Stop logging when the Sense HAT middle joystick button is pressed.
    """
    for event in sense.stick.get_events():
        if event.action == "pressed" and event.direction == "middle":
            return True
    return False


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
    sense.clear()

    dt_target = 1.0 / args.sample_rate_hz
    detector = StepDetector(
        threshold_g=args.threshold_g,
        min_interval_s=args.min_step_interval_s,
    )

    output_file = PROJECT_ROOT / "data" / "experiments" / f"sensor_log_walk_gyro11.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Logging to: {output_file}")
    print("Press the Sense HAT middle joystick button to stop.")
    print(
        f"Logger also stops automatically after {args.stop_after_inactive_s:.1f}s "
        "without a detected step.\n"
    )

    fieldnames = [
        "timestamp",
        "dt",
        "heading_rad",
        "accel_x",
        "accel_y",
        "accel_z",
        "accel_norm",
        "gyro_x_rads",
        "gyro_y_rads",
        "gyro_z_rads",
        "step_event",
    ]

    start_wall = time.time()
    prev_time = start_wall
    last_step_wall = start_wall
    n_rows = 0
    n_steps = 0
    last_status_print = start_wall
    stop_reason = "unknown"

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
                    last_step_wall = loop_start

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
                        f"t={timestamp:6.2f}s | rows={n_rows:5d} | "
                        f"steps={n_steps:3d} | accel_norm={accel_norm:.3f} | "
                        f"heading={heading_rad:.3f}"
                    )
                    last_status_print = loop_start

                # Manual stop via Sense HAT joystick
                if stop_requested_from_joystick(sense):
                    stop_reason = "manual_stop_joystick"
                    print("\nStopped by Sense HAT joystick.")
                    break

                # Automatic stop after inactivity
                if (loop_start - last_step_wall) >= args.stop_after_inactive_s:
                    stop_reason = "inactive_timeout"
                    print(
                        f"\nStopped automatically after "
                        f"{args.stop_after_inactive_s:.1f}s without a detected step."
                    )
                    break

                elapsed = time.time() - loop_start
                time.sleep(max(0.0, dt_target - elapsed))

        except KeyboardInterrupt:
            stop_reason = "keyboard_interrupt"
            print("\nStopped by user with Ctrl+C.")

        finally:
            sense.clear()

    print(f"Saved file: {output_file}")
    print(f"Total rows: {n_rows}")
    print(f"Detected steps: {n_steps}")
    print(f"Stop reason: {stop_reason}")


if __name__ == "__main__":
    main()