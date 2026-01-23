# sensor_pipeline.py
import time
import numpy as np
from sense_hat import SenseHat
import csv
import signal
import sys

# ===================== USER SETTINGS =====================
SAMPLE_RATE_HZ = 20                 # Requested: 20 Hz
LOG_FILENAME = "sensor_log_4.csv"   # Requested: easy to change name here
CALIB_SECONDS = 2.0                # Keep device still during this time
G_TO_MPS2 = 9.81                   # 1 g = 9.81 m/s^2

# ===================== 1) INITIALIZATION =====================
print("[INFO] Initializing Sensor Pipeline...")
sense = SenseHat()
sense.set_imu_config(True, True, False)  # gyro, accel ON; mag OFF
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

# ===================== 2) CORE PROCESSING CLASSES =====================
class Kalman1DAccelRandomWalk:
    """
    Optie A: 1D random-walk Kalman filter on acceleration.
    State x = a, model: a_k = a_{k-1} + w, measurement: z_k = a_k + v
    """
    def __init__(self, q=1e-3, r=(0.15 ** 2)):
        self.x = 0.0  # acceleration estimate
        self.P = 1.0  # estimate variance
        self.q = q    # process noise variance
        self.r = r    # measurement noise variance

    def update(self, z):
        # Predict
        self.P = self.P + self.q

        # Update
        K = self.P / (self.P + self.r)
        self.x = self.x + K * (z - self.x)
        self.P = (1.0 - K) * self.P
        return self.x

class SimpleStepDetector:
    """
    Detect steps + stationary based on dynamic acceleration magnitude.
    IMPORTANT: expects inputs in m/s^2.
    """
    def __init__(
        self,
        window_size=15,
        step_threshold_mps2=0.8 * G_TO_MPS2,
        stationary_var_threshold=(0.2 * G_TO_MPS2) ** 2,
        stationary_mean_threshold=0.3 * G_TO_MPS2,
    ):
        self.window_size = window_size
        self.step_threshold = step_threshold_mps2
        self.stat_var_threshold = stationary_var_threshold
        self.stat_mean_threshold = stationary_mean_threshold

        self.accel_history = []
        self.step_count = 0
        self.last_state = "MOVING"

    def add_reading(self, accel_vector_mps2, gravity_mps2):
        # Dynamic acceleration magnitude
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

        # Stationary: low variance and low mean dynamic accel
        if var < self.stat_var_threshold and abs(avg) < self.stat_mean_threshold:
            self.last_state = "STATIONARY"
            return "STATIONARY", self.step_count

        # Step: peak detection (simple)
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
        self.stationary_correction = True  # ZUPT-like: reset velocity when stationary

    def update(self, linear_accel_mps2, is_stationary=False, dt=None):
        if dt is None:
            now = time.time()
            dt = now - self.last_time
            self.last_time = now

        # Integrate
        self.velocity += linear_accel_mps2 * dt
        self.position += self.velocity * dt

        # Stationary -> reset velocity (requested point 3)
        if is_stationary and self.stationary_correction:
            self.velocity[:] = 0.0

        return self.velocity.copy(), self.position.copy(), float(dt)

# ===================== 3) MAIN PIPELINE =====================
def calibrate_gravity_bias(kf_x, kf_y, kf_z, seconds=2.0):
    """
    Estimate gravity+bias vector in sensor frame.
    User should keep device still during calibration.
    Returns g_est (m/s^2 vector).
    """
    print(f"[INFO] Calibrating gravity/bias for {seconds:.1f}s... Keep device still.")
    samples = []
    t0 = time.time()

    # Small sleep to avoid busy looping during calibration
    calib_sleep = 0.01

    while time.time() - t0 < seconds:
        accel_raw = sense.get_accelerometer_raw()  # in g
        accel_g = np.array([accel_raw["x"], accel_raw["y"], accel_raw["z"]], dtype=float)
        accel_mps2 = accel_g * G_TO_MPS2

        # filter once to reduce noise in mean estimate
        accel_filt = np.array([
            kf_x.update(accel_mps2[0]),
            kf_y.update(accel_mps2[1]),
            kf_z.update(accel_mps2[2]),
        ], dtype=float)

        samples.append(accel_filt)
        time.sleep(calib_sleep)

    g_est = np.mean(np.vstack(samples), axis=0)
    print(f"[INFO] g_est (gravity+bias) = [{g_est[0]:.3f}, {g_est[1]:.3f}, {g_est[2]:.3f}] m/s^2")
    return g_est

def main_pipeline(sample_rate_hz=20, log_filename="sensor_log_1.csv"):
    print(f"[INFO] Starting pipeline at {sample_rate_hz} Hz")
    print(f"[INFO] Logging to: {log_filename}")
    print("[INFO] Press Ctrl+C to stop.")

    # --- Modules ---
    kf_x = Kalman1DAccelRandomWalk()
    kf_y = Kalman1DAccelRandomWalk()
    kf_z = Kalman1DAccelRandomWalk()
    step_detector = SimpleStepDetector()
    integrator = VelocityIntegrator()

    # --- Calibration: gravity+bias estimate (IMPORTANT) ---
    g_est = calibrate_gravity_bias(kf_x, kf_y, kf_z, seconds=CALIB_SECONDS)

    # --- CSV header ---
    header = [
        "timestamp", "dt",
        "accel_raw_x_g", "accel_raw_y_g", "accel_raw_z_g",
        "accel_raw_x_mps2", "accel_raw_y_mps2", "accel_raw_z_mps2",
        "accel_filt_x_mps2", "accel_filt_y_mps2", "accel_filt_z_mps2",
        "accel_lin_x_mps2", "accel_lin_y_mps2", "accel_lin_z_mps2",
        "gyro_x", "gyro_y", "gyro_z",
        "step_state", "step_count",
        "velocity_x", "velocity_y", "velocity_z",
        "pos_x", "pos_y", "pos_z"
    ]

    target_dt = 1.0 / float(sample_rate_hz)

    with open(log_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        sample_count = 0
        start_time = time.time()

        while running:
            loop_start = time.time()
            sample_count += 1

            # ---- A) READ RAW SENSORS ----
            accel_raw = sense.get_accelerometer_raw()  # in g
            gyro_raw = sense.get_gyroscope_raw()       # rad/s (SenseHat convention)
            timestamp = time.time()

            accel_g = np.array([accel_raw["x"], accel_raw["y"], accel_raw["z"]], dtype=float)
            gyro = np.array([gyro_raw["x"], gyro_raw["y"], gyro_raw["z"]], dtype=float)

            # ---- B) UNITS FIX: g -> m/s^2 (requested point 2) ----
            accel_mps2 = accel_g * G_TO_MPS2

            # ---- C) KF (Optie A): accel random-walk KF (requested point 1) ----
            accel_filt = np.array([
                kf_x.update(accel_mps2[0]),
                kf_y.update(accel_mps2[1]),
                kf_z.update(accel_mps2[2]),
            ], dtype=float)

            # ---- D) Linear accel: remove gravity/bias estimate ----
            accel_lin = accel_filt - g_est

            # ---- E) STEP / STATIONARY ----
            step_state, step_count = step_detector.add_reading(accel_filt, gravity_mps2=g_est)
            is_stationary = (step_state == "STATIONARY")

            # ---- F) INTEGRATION + Stationary velocity reset (requested point 3) ----
            velocity, position, dt = integrator.update(accel_lin, is_stationary=is_stationary)

            # ---- G) LOG ----
            writer.writerow([
                timestamp, dt,
                accel_g[0], accel_g[1], accel_g[2],
                accel_mps2[0], accel_mps2[1], accel_mps2[2],
                accel_filt[0], accel_filt[1], accel_filt[2],
                accel_lin[0], accel_lin[1], accel_lin[2],
                gyro[0], gyro[1], gyro[2],
                step_state, step_count,
                velocity[0], velocity[1], velocity[2],
                position[0], position[1], position[2],
            ])

            # ---- H) LED FEEDBACK (lightweight) ----
            # Keep this minimal for performance
            if step_state == "STEP_DETECTED":
                sense.clear()
                if step_count <= 64:
                    idx = step_count - 1
                    row = idx // 8
                    col = idx % 8
                    sense.set_pixel(col, row, (0, 100, 0))
            elif step_state == "STATIONARY":
                # blink center pixel
                if int(time.time() * 2) % 2 == 0:
                    sense.set_pixel(3, 3, (100, 0, 0))
                else:
                    sense.set_pixel(3, 3, (0, 0, 0))

            # ---- I) CONSOLE STATUS (every ~2 seconds) ----
            if sample_count % max(int(sample_rate_hz * 2), 1) == 0:
                elapsed = time.time() - start_time
                print(f"[STATUS] t={elapsed:.1f}s | steps={step_count} | state={step_state} "
                      f"| pos=({position[0]:.2f}, {position[1]:.2f}) | dt={dt*1000:.1f}ms")

            # ---- J) RATE CONTROL ----
            processing_time = time.time() - loop_start
            sleep_time = target_dt - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sample_count % 50 == 0:
                print(f"[WARNING] Can't maintain {sample_rate_hz} Hz. "
                      f"Processing takes {processing_time:.3f}s")

    print(f"\n[INFO] Pipeline stopped. Data saved to {log_filename}")
    print(f"[INFO] Processed {sample_count} samples in {time.time()-start_time:.1f}s")
    sense.clear()

# ===================== 4) EXECUTION =====================
if __name__ == "__main__":
    main_pipeline(sample_rate_hz=SAMPLE_RATE_HZ, log_filename=LOG_FILENAME)
