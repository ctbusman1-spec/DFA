#!/usr/bin/env python3
# pi_min_log.py
#
# Sense HAT IMU -> CSV logger for DFA indoor positioning debugging.
# - STILL/MOVE: based on high-pass of (acc_mag - 1g) + gyro magnitude
# - STEP: peak detection on |high-pass|
# - HEADING: uses Sense HAT orientation yaw (sensor fusion) => turning works reliably
# - Position: step-based dead-reckoning with fixed stride
#
# Output CSV columns include yaw_rad and step_count etc.

import time
import math
import csv
from dataclasses import dataclass

from sense_hat import SenseHat


# =========================
# Optional: simple meter floorplan walkable flag (sanity only)
# =========================
@dataclass
class Rect:
    x0: float
    y0: float
    w: float
    h: float

    def contains(self, x: float, y: float) -> bool:
        return (self.x0 <= x <= self.x0 + self.w) and (self.y0 <= y <= self.y0 + self.h)


class NewFloorplanMeters:
    """
    Sanity walkable flag in meters (not the raster prior).
    Your laptop overlay uses the raster prior. This is just for logging.
    """
    def __init__(self):
        self.left_block = Rect(0.0, 0.0, 2.0, 2.3)
        self.right_corridor = Rect(2.0, 0.0, 5.0, 0.8)

        # 40cm drop fix
        vx0 = 2.0 + 5.0 - 0.8
        vy0 = 0.4
        vheight = 2.3
        self.vertical_corridor = Rect(vx0, vy0, 0.8, vheight)

        self.rects = [self.left_block, self.right_corridor, self.vertical_corridor]

    def is_walkable(self, x: float, y: float) -> bool:
        return any(r.contains(x, y) for r in self.rects)


# =========================
# Config
# =========================
@dataclass
class Config:
    LOG_SECONDS: float = 60.0
    TARGET_HZ: float = 32.0  # Pi tends to run ~31-32 Hz with Sense HAT calls

    # Start pose in meters (match your prior start)
    START_X_M: float = 1.0
    START_Y_M: float = 0.25

    # Fixed stride per detected step (tune later)
    STRIDE_M: float = 0.55

    # High-pass filter for motion/steps
    HP_ALPHA: float = 0.90  # 0.85–0.95

    # STILL detection
    STILL_HP_THR: float = 0.03
    STILL_GYRO_THR: float = 0.35  # rad/s (slightly looser than before)
    STILL_HOLD_S: float = 0.40

    # STEP detection (on |hp|)
    STEP_ON_THR: float = 0.08      # lowered a bit to catch indoor steps
    STEP_OFF_THR: float = 0.04
    STEP_MIN_INTERVAL_S: float = 0.30

    # Heading options
    USE_YAW_SENSORFUSION: bool = True  # recommended
    # If you ever want pure gyro integration instead:
    # GYRO_AXIS_FOR_YAW: str = "y"  # "x" / "y" / "z"
    WRAP_HEADING: bool = True


# =========================
# Detector
# =========================
class StepMoveDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.hp = 0.0

        self.in_peak = False
        self.last_step_t = -1e9

        self.is_still = True
        self.still_time = 0.0

    def update(self, t: float, dt: float, acc_mag_g: float, gyro_mag: float):
        # remove gravity (approx)
        lin = acc_mag_g - 1.0

        # cheap high-pass-ish
        self.hp = self.cfg.HP_ALPHA * (self.hp + lin)
        hp_abs = abs(self.hp)

        # STILL: quiet for some time
        if (hp_abs < self.cfg.STILL_HP_THR) and (gyro_mag < self.cfg.STILL_GYRO_THR):
            self.still_time += dt
        else:
            self.still_time = 0.0

        self.is_still = self.still_time >= self.cfg.STILL_HOLD_S

        # STEP: peak detection on hp_abs
        step = 0
        if not self.is_still:
            if (not self.in_peak) and (hp_abs > self.cfg.STEP_ON_THR) and ((t - self.last_step_t) > self.cfg.STEP_MIN_INTERVAL_S):
                self.in_peak = True

            if self.in_peak and (hp_abs < self.cfg.STEP_OFF_THR):
                self.in_peak = False
                self.last_step_t = t
                step = 1
        else:
            self.in_peak = False

        state = "STILL" if self.is_still else "MOVE"
        return state, step, self.hp


# =========================
# Utils
# =========================
def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def main():
    cfg = Config()
    hat = SenseHat()
    fp = NewFloorplanMeters()
    det = StepMoveDetector(cfg)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = f"imu_log_4.csv"

    print(f"[INFO] Logging -> {out_file}")
    print(f"[INFO] Duration: {cfg.LOG_SECONDS}s | Target: {cfg.TARGET_HZ}Hz | Stride: {cfg.STRIDE_M}m")
    print("[INFO] Protocol: 5s still -> walk -> 5s still")

    # state
    x = cfg.START_X_M
    y = cfg.START_Y_M
    step_count = 0

    # timing
    t0 = time.time()
    last = t0
    dts = []

    # heading
    heading = 0.0

    with open(out_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t", "dt",
            "acc_x_g", "acc_y_g", "acc_z_g", "acc_mag_g",
            "gyro_x_rads", "gyro_y_rads", "gyro_z_rads", "gyro_mag_rads",
            "yaw_deg", "yaw_rad",
            "hp", "state", "step", "step_count",
            "heading_rad",
            "pos_x_m", "pos_y_m",
            "walkable"
        ])

        while True:
            now = time.time()
            t = now - t0
            if t >= cfg.LOG_SECONDS:
                break

            dt = now - last
            last = now
            if dt <= 0:
                dt = 1e-6
            dts.append(dt)

            # read sensors
            a = hat.get_accelerometer_raw()   # g
            g = hat.get_gyroscope_raw()       # rad/s
            ori = hat.get_orientation()       # degrees (yaw/pitch/roll)

            acc_x = float(a["x"])
            acc_y = float(a["y"])
            acc_z = float(a["z"])
            acc_mag = math.sqrt(acc_x*acc_x + acc_y*acc_y + acc_z*acc_z)

            gx = float(g["x"])
            gy = float(g["y"])
            gz = float(g["z"])
            gyro_mag = math.sqrt(gx*gx + gy*gy + gz*gz)

            yaw_deg = float(ori["yaw"])
            yaw_rad = math.radians(yaw_deg)

            # heading source
            if cfg.USE_YAW_SENSORFUSION:
                heading = yaw_rad
            else:
                # fallback: integrate one axis (not recommended)
                # heading += gy * dt
                heading = heading

            if cfg.WRAP_HEADING:
                heading = wrap_pi(heading)

            # detect
            state, step, hp = det.update(t, dt, acc_mag, gyro_mag)

            # update position on steps
            if step == 1:
                step_count += 1
                x += cfg.STRIDE_M * math.cos(heading)
                y += cfg.STRIDE_M * math.sin(heading)

            walkable = 1 if fp.is_walkable(x, y) else 0

            w.writerow([
                f"{t:.6f}", f"{dt:.6f}",
                f"{acc_x:.6f}", f"{acc_y:.6f}", f"{acc_z:.6f}", f"{acc_mag:.6f}",
                f"{gx:.6f}", f"{gy:.6f}", f"{gz:.6f}", f"{gyro_mag:.6f}",
                f"{yaw_deg:.3f}", f"{yaw_rad:.6f}",
                f"{hp:.6f}", state, step, step_count,
                f"{heading:.6f}",
                f"{x:.6f}", f"{y:.6f}",
                walkable
            ])

            # sleep toward target
            target_dt = 1.0 / max(1.0, cfg.TARGET_HZ)
            loop_time = time.time() - now
            time.sleep(max(0.0, target_dt - loop_time))

    # summary
    hz_eff = 1.0 / (sum(dts) / len(dts)) if len(dts) > 10 else float("nan")
    print("[DONE] Log complete.")
    print(f"[DONE] Steps counted: {step_count}")
    print(f"[DONE] Effective Hz: {hz_eff:.1f}")
    print(f"[DONE] File: {out_file}")
    print(f"[TIP] Copy to laptop: scp pi@<PI_IP>:{out_file} .")


if __name__ == "__main__":
    main()
