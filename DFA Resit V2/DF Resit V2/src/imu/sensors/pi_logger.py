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


def safe_float(value, default: float = 0.0) -> float:
    try:
        v = float(value)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default


def main():
    if SenseHat is None:
        raise RuntimeError("sense_hat is not available on this machine.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pi_sensor_log")
    parser.add_argument("--sample-rate-hz", type=int, default=50)
    parser.add_argument("--threshold-g", type=float, default=1.18)
    parser.add_argument("--min-step-interval-s", type=float, default=0.30)
    parser.add_argument("--stop-after-inactive-s", type=float, default=10.0)
    parser.add_argument("--timestamped-name", action="store_true")
    args = parser.parse_args()

    sense = SenseHat()
    sense.clear()

    dt_target = 1.0 / max(args.sample_rate_hz, 1)
    detector = StepDetector(
        threshold_g=args.threshold_g,
        min_interval_s=args.min_step_interval_s,
    )

    file_stem = args.name
    if args.timestamped_name:
        file_stem = f"{args.name}_{time.strftime('%Y%m%d_%H%M%S')}"

    output_file = PROJECT_ROOT / "data" / "experiments" / (f"sensor_log_gyro111"
                                                           f".csv")
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

                if not math.isfinite(dt) or dt <= 0:
                    dt = dt_target

                # Accelerometer
                acc = sense.get_accelerometer_raw()
                ax = safe_float(acc.get("x", 0.0))
                ay = safe_float(acc.get("y", 0.0))
                az = safe_float(acc.get("z", 0.0))

                # Orientation
                ori = sense.get_orientation_radians()
                heading_rad = safe_float(ori.get("yaw", 0.0))

                # Gyroscope raw
                gyro = sense.get_gyroscope_raw()

                gx = safe_float(gyro.get("x", 0.0))
                gy = safe_float(gyro.get("y", 0.0))
                gz = safe_float(gyro.get("z", 0.0))

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
                    "gyro_x_rads": round(gx, 6),
                    "gyro_y_rads": round(gy, 6),
                    "gyro_z_rads": round(gz, 6),
                    "step_event": int(step_event),
                })
                f.flush()
                n_rows += 1

                if loop_start - last_status_print >= 1.0:
                    print(
                        f"t={timestamp:6.2f}s | rows={n_rows:5d} | "
                        f"steps={n_steps:3d} | accel_norm={accel_norm:.3f} | "
                        f"heading={heading_rad:.3f} | gz={gz:.3f} rad/s"
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