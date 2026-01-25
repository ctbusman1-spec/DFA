#!/usr/bin/env python3
# pi_min_log.py
# Minimal IMU logger (Sense HAT) -> CSV
# Logs: raw accel (g), raw gyro (rad/s), still/move state, step events, heading, dead-reckoning pos,
# AND floorplan "walkable" flag based on your new floorplan geometry.

import time
import math
import csv
from dataclasses import dataclass

import numpy as np
from sense_hat import SenseHat


# =========================
# Floorplan (NEW, fixed)
# =========================
# Coordinate system: x to the right (m), y downward (m) for plotting convenience.
# Walkable area = union of filled rectangles (no holes).
# - Left block: 2.0 x 2.3
# - Right corridor: 5.0 x 0.8
# - Vertical corridor: width 0.8, starts 0.4 m lower than the previous version (important fix)
#
# We implement a simple point-in-rect walkable(x,y) check here (fast, Pi-friendly).

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
        # You can tweak origin placement here if your plotting expects a different reference.
        # We'll place the left block with its top-left corner at (0,0).
        self.left_block = Rect(0.0, 0.0, 2.0, 2.3)

        # Horizontal corridor (right): starts at right edge of left block, aligned at y = 0.0
        # width=5.0 length (x direction), height=0.8
        self.right_corridor = Rect(2.0, 0.0, 5.0, 0.8)

        # Vertical corridor: width=0.8, starts 0.4 m lower than before
        # We connect it to the right corridor near its far right end.
        # Put it at x = 2.0 + 5.0 - 0.8 (so it attaches at the corridor end, centered)
        vx0 = 2.0 + 5.0 - 0.8
        vy0 = 0.4  # <-- the "40 cm lower" fix
        # Choose a reasonable vertical corridor height; adjust if your map has a known exact value.
        # If you know the exact height, replace this.
        vheight = 2.3  # works nicely with left block height
        self.vertical_corridor = Rect(vx0, vy0, 0.8, vheight)

        self.rects = [self.left_block, self.right_corridor, self.vertical_corridor]

    def is_walkable(self, x: float, y: float) -> bool:
        return any(r.contains(x, y) for r in self.rects)


# =========================
# Simple motion / step logic
# =========================

@dataclass
class Config:
    hz: float = 50.0
    log_seconds: float = 60.0

    # still detection (acc magnitude close to 1g + low gyro)
    still_acc_tol_g: float = 0.05
    still_gyro_tol_rads: float = 0.20
    still_hold_s: float = 0.40  # how long thresholds must be met to declare STILL

    # step detection on filtered acc magnitude "bump"
    step_min_interval_s: float = 0.28
    step_threshold_g: float = 0.18  # peak above baseline (tune)
    step_hysteresis_g: float = 0.08

    # stride model (fixed stride length per step for now)
    stride_m: float = 0.70

    # start position as per your description
    start_x_m: float = 1.0
    start_y_m: float = 0.25


class StepMoveDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.baseline_g = 1.0
        self.alpha = 0.02  # baseline low-pass

        self.in_peak = False
        self.last_step_t = -1e9

        self.still_counter = 0
        self.is_still = True

    def update(self, t: float, acc_mag_g: float, gyro_z_rads: float) -> tuple[str, int]:
        # Update baseline slowly
        self.baseline_g = (1 - self.alpha) * self.baseline_g + self.alpha * acc_mag_g
        bump = acc_mag_g - self.baseline_g

        # STILL logic
        acc_ok = abs(acc_mag_g - 1.0) <= self.cfg.still_acc_tol_g
        gyro_ok = abs(gyro_z_rads) <= self.cfg.still_gyro_tol_rads

        if acc_ok and gyro_ok:
            self.still_counter += 1
        else:
            self.still_counter = 0

        still_needed = int(self.cfg.still_hold_s * self.cfg.hz)
        prev_still = self.is_still
        self.is_still = self.still_counter >= still_needed

        # STEP detection (only when not still)
        step_event = 0
        if not self.is_still:
            # rising above threshold -> enter peak
            if (not self.in_peak) and (bump >= self.cfg.step_threshold_g) and (t - self.last_step_t >= self.cfg.step_min_interval_s):
                self.in_peak = True
            # falling below hysteresis -> confirm step
            if self.in_peak and (bump <= self.cfg.step_hysteresis_g):
                self.in_peak = False
                self.last_step_t = t
                step_event = 1

        # State label
        if self.is_still:
            state = "STILL"
        else:
            state = "MOVE"

        # If transition still->move or move->still, you’ll see it in the state stream.
        # Step is a separate binary flag.
        return state, step_event


def main():
    cfg = Config(hz=32.0)
    hat = SenseHat()
    fp = NewFloorplan()
    det = StepMoveDetector(cfg)

    dt = 1.0 / cfg.hz
    n = int(cfg.log_seconds * cfg.hz)

    # Heading + position
    heading = 0.0  # radians
    x = cfg.start_x_m
    y = cfg.start_y_m

    # CSV file
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = f"imu_log_3.csv"
    print(f"[INFO] Logging to {out_file} for {cfg.log_seconds}s @ {cfg.hz}Hz")

    with open(out_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t", "dt",
            "acc_x_g", "acc_y_g", "acc_z_g", "acc_mag_g",
            "gyro_x_rads", "gyro_y_rads", "gyro_z_rads",
            "state", "step",
            "heading_rad",
            "pos_x_m", "pos_y_m",
            "walkable"
        ])

        t0 = time.time()
        last_t = t0

        for i in range(n):
            now = time.time()
            t = now - t0
            dti = now - last_t
            last_t = now

            # Sense HAT raw data
            a = hat.get_accelerometer_raw()  # in g
            g = hat.get_gyroscope_raw()      # in rad/s (Sense HAT returns radians/sec)

            acc_x = float(a["x"])
            acc_y = float(a["y"])
            acc_z = float(a["z"])
            acc_mag = math.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

            gyro_x = float(g["x"])
            gyro_y = float(g["y"])
            gyro_z = float(g["z"])

            # Update heading (integrate gyro z)
            heading += gyro_z * dti
            # wrap to [-pi, pi]
            heading = (heading + math.pi) % (2 * math.pi) - math.pi

            # Detect movement + step
            state, step = det.update(t, acc_mag, gyro_z)

            # Dead-reckoning: update position on step
            if step == 1:
                stride = cfg.stride_m
                x += stride * math.cos(heading)
                y += stride * math.sin(heading)

            walkable = 1 if fp.is_walkable(x, y) else 0

            w.writerow([
                f"{t:.6f}", f"{dti:.6f}",
                f"{acc_x:.6f}", f"{acc_y:.6f}", f"{acc_z:.6f}", f"{acc_mag:.6f}",
                f"{gyro_x:.6f}", f"{gyro_y:.6f}", f"{gyro_z:.6f}",
                state, step,
                f"{heading:.6f}",
                f"{x:.6f}", f"{y:.6f}",
                walkable
            ])

            # Sleep to target rate
            # (Use max(0, ...) to avoid negative if loop runs long)
            elapsed = time.time() - now
            time.sleep(max(0.0, dt - elapsed))

    print("[DONE] Log complete.")
    print(f"[TIP] Copy CSV to laptop, e.g.: scp pi@<PI_IP>:{out_file} .")


if __name__ == "__main__":
    main()
