# sensor_pipeline.py
# - 20 Hz logging
# - Accel random-walk KF (Optie A)
# - Units: g -> m/s^2
# - Stationary -> velocity reset
# - Baseline experiment helper: run markers + simple countdowns
# - FIX: reset integrator timing after calibration (prevents huge first dt)

import time
import numpy as np
from sense_hat import SenseHat
import csv
import signal
import sys

# ===================== USER SETTINGS =====================
SAMPLE_RATE_HZ = 20
LOG_FILENAME = "sensor_log_3.csv"   # Change this easily
CALIB_SECONDS = 2.0                # Keep device still during this time
COUNTDOWN_SECONDS = 3              # 3s still before & after run (helps segmentation)
G_TO_MPS2 = 9.81

# Step/stationary parameters (tuned for "walking"; OK for baseline but may show mostly STATIONARY)
WINDOW_SIZE = 15
STEP_THRESHOLD_MPS2 = 0.8 * G_TO_MPS2
STATIONARY_VAR_THRESHOLD = (0.2 * G_TO_MPS2) ** 2
STATIONARY_MEAN_THRESHOLD = 0.3 * G_TO_MPS2

# ===================== INIT =====================
print("[INFO] Initializing Sensor Pipeline...")
sense = SenseHat()
sense.set_imu_config(True, True, False)  # gyro+accel ON, mag OFF
sense.clear()

running = True
def signal_handler(sig, frame):
    global running
    print("\n[INFO] Shutdown signal received. Stopping...")
    running = False
    try:
        sense.clear()
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ===================== MODULES =====================
class Kalman1DAccelRandomWalk:
    """Optie A: 1D random-walk KF on acceleration (m/s^2)."""
    def __init__(self, q=1e-3, r=(0.15 ** 2)):
        self.x = 0.0
        self.P = 1.0
        self.q = q
        self.r = r

    def update(self, z):
        self.P = self.P + self.q
        K = self.P / (self.P + self.r)
        self.x = self.x + K * (z - self.x)
        self.P = (1.0 - K) * self.P
        return self.x

class SimpleStepDetector:
    """Detects steps and stationary via dynamic accel magnitude. Expects m/s^2."""
    def __init__(self, window_size=WINDOW_SIZE,
                 step_threshold_mps2=STEP_THRESHOLD_MPS2,
                 stationary_var_threshold=STATIONARY_VAR_THRESHOLD,
                 stationary_mean_threshold=STATIONARY_MEAN_THRESHOLD):
        self.window_size = window_size
        self.step_threshold = step_threshold_mps2
        self.stat_var_threshold = stationary_var_threshold
        self.stat_mean_threshold = stationary_mean_threshold
        self.accel_history = []
        self.step_count = 0
        self.last_state = "MOVING"

    def add_reading(self, accel_vector_mps2, gravity_mps2):
        dynamic = accel_vector_mps2 - gravity_mps2
        mag = float(np.linalg.norm(dynamic))
        self.accel_history.append(mag)
        if len(self.accel_history) > self.window_size:
            self.accel_history.pop(0)
        return self._analyze()

    def _analyze(self):
        if len(self.accel_history) < self.window_size:
            return "INITIALIZING", 0

        recent = np.array(self.accel_history[-5:], dtype=float)
        avg = float(np.mean(self.accel_history))
        var = float(np.var(recent))

        if var < self.stat_var_threshold and abs(avg) < self.stat_mean_threshold:
            self.last_state = "STATIONARY"
            return "STATIONARY", self.step_count

        if np.max(recent) > self.step_threshold and self.last_state != "STEP_DETECTED":
            self.step_count += 1
            self.last_state = "STEP_DETECTED"
            return "STEP_DETECTED", self.step_count

        self.last_state = "MOVING"
        return "MOVING", self.step_count

class VelocityIntegrator:
    """Integrates linear acceleration (m/s^2) to velocity and position."""
    def __init__(self):
        self.velocity = np.zeros(3, dtype=float)
        self.position = np.zeros(3, dtype=float)
        self.last_time = time.time()
        self.stationary_correction = True

    def reset(self):
        self.velocity[:] = 0.0
        self.position[:] = 0.0
        self.last_time = time.time()

    def update(self, linear_accel_mps2, is_stationary=False):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        self.velocity += linear_accel_mps2 * dt
        self.position += self.velocity * dt

        if is_stationary and self.stationary_correction:
            self.velocity[:] = 0.0

        return self.velocity.copy(), self.position.copy(), float(dt)

def countdown(msg, seconds=3):
    print(msg)
    for i in range(seconds, 0, -1):
        print(f"  {i}...")
        time.sleep(1.0)

def calibrate_gravity_bias(kf_x, kf_y, kf_z, seconds=2.0):
    """Estimate gravity+bias vector in sensor frame (m/s^2). Keep device still."""
    print(f"[INFO] Calibrating gravity/bias for {seconds:.1f}s... keep device STILL.")
    samples = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        accel_raw = sense.get_accelerometer_raw()  # g
        accel_g = np.array([accel_raw["x"], accel_raw["y"], accel_raw["z"]], dtype=float)
        accel_mps2 = accel_g * G_TO_MPS2

        accel_filt = np.array([
            kf_x.update(accel_mps2[0]),
            kf_y.update(accel_mps2[1]),
            kf_z.update(accel_mps2[2]),
        ], dtype=float)

        samples.append(accel_filt)
        time.sleep(0.01)

    g_est = np.mean(np.vstack(samples), axis=0)
    print(f"[INFO] g_est = [{g_est[0]:.3f}, {g_est[1]:.3f}, {g_est[2]:.3f}] m/s^2")
    return g_est

# ===================== PIPELINE =====================
def main_pipeline(sample_rate_hz=SAMPLE_RATE_HZ, log_filename=LOG_FILENAME):
    print(f"[INFO] Starting pipeline at {sample_rate_hz} Hz")
    print(f"[INFO] Logging to: {log_filename}")
    print("[INFO] Press Ctrl+C to stop at any time.\n")

    # Modules
    kf_x = Kalman1DAccelRandomWalk()
    kf_y = Kalman1DAccelRandomWalk()
    kf_z = Kalman1DAccelRandomWalk()
    step_detector = SimpleStepDetector()
    integrator = VelocityIntegrator()

    # Calibration (still)
    countdown("[EXPERIMENT] Put the device on the table in the SAME orientation you will start. Stay still.", COUNTDOWN_SECONDS)
    g_est = calibrate_gravity_bias(kf_x, kf_y, kf_z, seconds=CALIB_SECONDS)

    # IMPORTANT: reset integrator after calibration (prevents huge first dt)
    integrator.reset()

    # Experiment markers (manual protocol)
    print("\n[EXPERIMENT PROTOCOL]")
    print("1) Still 3s (marker START_STILL)")
    print("2) Move back-and-forth along 1.00 m line for your chosen passes (marker RUN_MOVING)")
    print("3) Still 3s (marker END_STILL)")
    print("Then STOP the program (Ctrl+C), rotate 90 degrees, change LOG_FILENAME, repeat.\n")

    countdown("[EXPERIMENT] START_STILL: keep device still.", COUNTDOWN_SECONDS)
    print("[EXPERIMENT] RUN_MOVING: start moving now (back-and-forth over 1 m).")
    print("           When done, stop moving and hold still; the log will capture END_STILL.\n")

    # CSV header
    header = [
        "timestamp", "dt",
        "phase",  # START_STILL / RUN_MOVING / END_STILL (best-effort)
        "accel_raw_x_g", "accel_raw_y_g", "accel_raw_z_g",
        "accel_raw_x_mps2", "accel_raw_y_mps2", "accel_raw_z_mps2",
        "accel_filt_x_mps2", "accel_filt_y_mps2", "accel_filt_z_mps2",
        "accel_lin_x_mps2", "accel_lin_y_mps2", "accel_lin_z_mps2",
        "gyro_x", "gyro_y", "gyro_z",
        "step_state", "step_count",
        "velocity_x", "velocity_y", "velocity_z",
        "pos_x", "pos_y", "pos_z",
        "g_est_x", "g_est_y", "g_est_z"
    ]

    target_dt = 1.0 / float(sample_rate_hz)
    start_time = time.time()

    # We log phases by time since start:
    # first COUNTDOWN_SECONDS seconds = START_STILL
    # then until user stops moving (detected by STATIONARY for a while) = RUN_MOVING
    # after we've seen stationary consistently for ~1.5s after moving = END_STILL
    moving_started = False
    end_still_triggered = False
    stationary_streak = 0
    stationary_needed = int(1.5 * sample_rate_hz)  # ~1.5 seconds of stationary to call END_STILL

    with open(log_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        sample_count = 0

        while running:
            loop_start = time.time()
            sample_count += 1

            # ---- Read sensors ----
            accel_raw = sense.get_accelerometer_raw()
            gyro_raw = sense.get_gyroscope_raw()
            timestamp = time.time()

            accel_g = np.array([accel_raw["x"], accel_raw["y"], accel_raw["z"]], dtype=float)
            gyro = np.array([gyro_raw["x"], gyro_raw["y"], gyro_raw["z"]], dtype=float)

            # ---- Units: g -> m/s^2 ----
            accel_mps2 = accel_g * G_TO_MPS2

            # ---- KF (Optie A) ----
            accel_filt = np.array([
                kf_x.update(accel_mps2[0]),
                kf_y.update(accel_mps2[1]),
                kf_z.update(accel_mps2[2]),
            ], dtype=float)

            # ---- Linear accel using fixed gravity estimate (baseline) ----
            accel_lin = accel_filt - g_est

            # ---- Step / stationary ----
            step_state, step_count = step_detector.add_reading(accel_filt, gravity_mps2=g_est)
            is_stationary = (step_state == "STATIONARY")

            # ---- Integrate + stationary reset ----
            velocity, position, dt = integrator.update(accel_lin, is_stationary=is_stationary)

            # ---- Phase labeling (best-effort) ----
            t_rel = timestamp - start_time
            if t_rel < COUNTDOWN_SECONDS:
                phase = "START_STILL"
            else:
                if not moving_started:
                    # first time we see MOVING or STEP_DETECTED after initial still
                    if step_state in ("MOVING", "STEP_DETECTED"):
                        moving_started = True
                    phase = "RUN_MOVING" if moving_started else "START_STILL"
                else:
                    phase = "RUN_MOVING"
                    if is_stationary:
                        stationary_streak += 1
                    else:
                        stationary_streak = 0
                    if (not end_still_triggered) and stationary_streak >= stationary_needed:
                        end_still_triggered = True
                    if end_still_triggered:
                        phase = "END_STILL"

            # ---- Log ----
            writer.writerow([
                timestamp, dt, phase,
                accel_g[0], accel_g[1], accel_g[2],
                accel_mps2[0], accel_mps2[1], accel_mps2[2],
                accel_filt[0], accel_filt[1], accel_filt[2],
                accel_lin[0], accel_lin[1], accel_lin[2],
                gyro[0], gyro[1], gyro[2],
                step_state, step_count,
                velocity[0], velocity[1], velocity[2],
                position[0], position[1], position[2],
                g_est[0], g_est[1], g_est[2],
            ])

            # ---- Lightweight LED feedback ----
            if step_state == "STEP_DETECTED":
                sense.clear()
                if step_count <= 64:
                    idx = step_count - 1
                    row = idx // 8
                    col = idx % 8
                    sense.set_pixel(col, row, (0, 100, 0))
            elif step_state == "STATIONARY":
                if int(time.time() * 2) % 2 == 0:
                    sense.set_pixel(3, 3, (100, 0, 0))
                else:
                    sense.set_pixel(3, 3, (0, 0, 0))

            # ---- Status every ~2s ----
            if sample_count % max(int(sample_rate_hz * 2), 1) == 0:
                elapsed = time.time() - start_time
                print(f"[STATUS] t={elapsed:.1f}s | phase={phase} | state={step_state} | "
                      f"pos=({position[0]:.2f},{position[1]:.2f}) | dt={dt*1000:.1f}ms")

            # ---- Rate control ----
            processing_time = time.time() - loop_start
            sleep_time = target_dt - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    print(f"\n[INFO] Saved: {log_filename}")
    sense.clear()

# ===================== RUN =====================
if __name__ == "__main__":
    main_pipeline()
