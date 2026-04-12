#!/usr/bin/env python3
# pi_mu_logger.py
# DFA IMU logger for indoor pedestrian DR debugging.
# Use gyro-yaw integration with bias correction learned during STILL so laptop and screen on the desk won't interfere
# Output: CSV with accel/gyro, hp motion signal, state, step, heading, pos.

import time
import math
import csv
from dataclasses import dataclass
from collections import deque

from sense_hat import SenseHat


# keeps heading between -pi and pi
def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# checks if point falls within the floorplan
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
    creates floorplan meter object
    """
    def __init__(self):
        self.left_block = Rect(0.0, 0.0, 2.0, 2.3)
        self.right_corridor = Rect(2.0, 0.0, 5.0, 0.8)
        vx0 = 2.0 + 5.0 - 0.8
        vy0 = 0.4
        vheight = 2.3
        self.vertical_corridor = Rect(vx0, vy0, 0.8, vheight)
        self.rects = [self.left_block, self.right_corridor, self.vertical_corridor]

    def is_walkable(self, x: float, y: float) -> bool:
        return any(r.contains(x, y) for r in self.rects)



# Config
@dataclass
class Config:
    LOG_SECONDS: float = 40.0
    TARGET_HZ: float = 32.0

    # Start pose
    START_X_M: float = 1.0
    START_Y_M: float = 0.7

    # Stride length
    STRIDE_M: float = 0.45

    # Motion HP filter (for steps/still)
    HP_ALPHA: float = 0.90

    # STILL detection
    STILL_HP_THR: float = 0.03
    STILL_GYRO_THR: float = 0.35   # rad/s magnitude
    STILL_HOLD_S: float = 0.50

    # STEP detection (on |hp|)
    STEP_ON_THR: float = 0.08
    STEP_OFF_THR: float = 0.04
    STEP_MIN_INTERVAL_S: float = 0.30

    # Gyro yaw integration
    # IMPORTANT: choose the axis that corresponds to yaw
    YAW_AXIS: str = "z"

    # Bias learning
    BIAS_WINDOW_S: float = 2.0      # window length for bias estimation while STILL
    BIAS_MIN_STILL_S: float = 1.0   # need this much STILL time before using bias confidently

    # Map offset (radians)
    MAP_YAW_OFFSET_RAD: float = 0.0


# Detector (STILL/MOVE + STEP)
class StepMoveDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.hp = 0.0
        self.in_peak = False
        self.last_step_t = -1e9
        self.is_still = True
        self.still_time = 0.0

    def update(self, t: float, dt: float, acc_mag_g: float, gyro_mag: float):
        # Remove gravity
        lin = acc_mag_g - 1.0

        # High-pass-ish signal
        self.hp = self.cfg.HP_ALPHA * (self.hp + lin)
        hp_abs = abs(self.hp)

        # STILL logic
        if (hp_abs < self.cfg.STILL_HP_THR) and (gyro_mag < self.cfg.STILL_GYRO_THR):
            self.still_time += dt
        else:
            self.still_time = 0.0

        self.is_still = self.still_time >= self.cfg.STILL_HOLD_S

        # STEP logic (hysteresis peak)
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



# Gyro yaw integrator with bias learning
class GyroYaw:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.heading = 0.0
        self.bias = 0.0

        self.still_accum = 0.0
        self.bias_samples = deque()  # (t, yaw_rate)

    def _select_axis(self, gx, gy, gz):
        ax = self.cfg.YAW_AXIS.lower()
        if ax == "x":
            return gx
        if ax == "y":
            return gy
        return gz

    def update(self, t: float, dt: float, gx: float, gy: float, gz: float, is_still: bool):
        yaw_rate = self._select_axis(gx, gy, gz)

        # When STILL: collect samples to estimate bias
        if is_still:
            self.still_accum += dt
            self.bias_samples.append((t, yaw_rate))

            # drop old samples beyond window
            window = self.cfg.BIAS_WINDOW_S
            while self.bias_samples and (t - self.bias_samples[0][0] > window):
                self.bias_samples.popleft()

            # update bias when we have enough still time
            if self.still_accum >= self.cfg.BIAS_MIN_STILL_S and len(self.bias_samples) >= 5:
                self.bias = sum(v for _, v in self.bias_samples) / len(self.bias_samples)
        else:
            self.still_accum = 0.0
            # keep bias_samples as last window (helps stability)

        # integrate
        self.heading = wrap_pi(self.heading + (yaw_rate - self.bias) * dt)
        return self.heading, yaw_rate, self.bias



# Main
def main():
    cfg = Config()
    hat = SenseHat()
    fp = NewFloorplanMeters()

    det = StepMoveDetector(cfg)
    yaw = GyroYaw(cfg)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = f"imu_log_79.csv"

    print(f"Logging -> {out_file}")
    print(f"LOG_SECONDS={cfg.LOG_SECONDS} | TARGET_HZ={cfg.TARGET_HZ} | STRIDE={cfg.STRIDE_M}")
    print(f"YAW_AXIS='{cfg.YAW_AXIS}' | MAP_YAW_OFFSET_RAD={cfg.MAP_YAW_OFFSET_RAD:.3f}")
    print("Protocol: put Pi on desk, run, keep STILL 5s, then walk, then STILL 5s.")

    # state
    x = cfg.START_X_M
    y_pos = cfg.START_Y_M
    step_count = 0

    t0 = time.time()
    last = t0
    dts = []

    with open(out_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t", "dt",
            "acc_x_g", "acc_y_g", "acc_z_g", "acc_mag_g",
            "gyro_x_rads", "gyro_y_rads", "gyro_z_rads", "gyro_mag_rads",
            "yaw_rate_used", "gyro_bias",
            "hp", "state", "step", "step_count",
            "heading_rad",
            "pos_x_m", "pos_y_m",
            "walkable",
            # optional debug: sensor fusion yaw (not used)
            "sf_yaw_deg"
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
            a = hat.get_accelerometer_raw()  # g
            g = hat.get_gyroscope_raw()      # rad/s
            # sensor fusion yaw for debug only
            ori = hat.get_orientation()
            sf_yaw_deg = float(ori["yaw"])

            ax = float(a["x"]); ay = float(a["y"]); az = float(a["z"])
            acc_mag = math.sqrt(ax*ax + ay*ay + az*az)

            gx = float(g["x"]); gy = float(g["y"]); gz = float(g["z"])
            gyro_mag = math.sqrt(gx*gx + gy*gy + gz*gz)

            # detect motion & steps
            state, step, hp = det.update(t, dt, acc_mag, gyro_mag)

            # heading from gyro integration (+ bias)
            heading, yaw_rate_used, bias = yaw.update(t, dt, gx, gy, gz, det.is_still)

            # rotate into map frame
            heading_map = wrap_pi(heading + cfg.MAP_YAW_OFFSET_RAD)

            # step-based position update
            if step == 1:
                step_count += 1
                x += cfg.STRIDE_M * math.cos(heading_map)
                y_pos += cfg.STRIDE_M * math.sin(heading_map)

            walkable = 1 if fp.is_walkable(x, y_pos) else 0

            w.writerow([
                f"{t:.6f}", f"{dt:.6f}",
                f"{ax:.6f}", f"{ay:.6f}", f"{az:.6f}", f"{acc_mag:.6f}",
                f"{gx:.6f}", f"{gy:.6f}", f"{gz:.6f}", f"{gyro_mag:.6f}",
                f"{yaw_rate_used:.6f}", f"{bias:.6f}",
                f"{hp:.6f}", state, step, step_count,
                f"{heading_map:.6f}",
                f"{x:.6f}", f"{y_pos:.6f}",
                walkable,
                f"{sf_yaw_deg:.3f}",
            ])

            # sleep toward target Hz
            target_dt = 1.0 / max(1.0, cfg.TARGET_HZ)
            loop_time = time.time() - now
            time.sleep(max(0.0, target_dt - loop_time))

    hz_eff = 1.0 / (sum(dts) / len(dts)) if len(dts) > 10 else float("nan")

    print("Log complete.")
    print(f"Steps counted: {step_count}")
    print(f"Effective Hz: {hz_eff:.1f}")
    print(f"File: {out_file}")


if __name__ == "__main__":
    main()
