# sensor_pipeline.py
import time
import numpy as np
from sense_hat import SenseHat
import csv
from datetime import datetime
import signal
import sys

G_TO_MPS2 = 9.81  # 1 g = 9.81 m/s^2

# ========== 1. INITIALIZATION ==========
print("[INFO] Initializing Sensor Pipeline...")
sense = SenseHat()
sense.set_imu_config(True, True, False)  # Enable gyro, accel, disable mag
sense.clear()

running = True
def signal_handler(sig, frame):
    global running
    print('[INFO] Shutdown signal received.')
    running = False
    sense.clear()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ========== 2. CORE PROCESSING CLASSES ==========
class Kalman1DAccelRandomWalk:
    """
    Optie A: 1D Random-walk Kalman Filter op acceleratie.
    State x = a (acceleratie), model: a_k = a_{k-1} + w, z_k = a_k + v
    """
    def __init__(self, q=1e-3, r=(0.15**2)):
        self.x = 0.0   # estimated acceleration
        self.P = 1.0   # estimated variance
        self.q = q     # process noise variance
        self.r = r     # measurement noise variance

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
    Detects steps and stationary phases using accelerometer magnitude.
    Let op: hier werken we nu in m/s^2 (dus thresholds zijn ook m/s^2).
    """
    def __init__(self, window_size=15,
                 step_threshold_mps2=0.8 * G_TO_MPS2,
                 stationary_var_threshold=0.2 * (G_TO_MPS2**2),
                 stationary_mean_threshold=0.3 * G_TO_MPS2):
        self.window_size = window_size
        self.step_threshold = step_threshold_mps2
        self.stat_var_threshold = stationary_var_threshold
        self.stat_mean_threshold = stationary_mean_threshold

        self.accel_history = []
        self.step_count = 0
        self.last_state = "MOVING"

    def add_reading(self, accel_vector_mps2, gravity_mps2=np.array([0, 0, G_TO_MPS2])):
        # Gravity removal (simpel; werkt het best als de IMU ongeveer upright is)
        dynamic_accel = accel_vector_mps2 - gravity_mps2
        accel_magnitude = np.linalg.norm(dynamic_accel)
        self.accel_history.append(accel_magnitude)

        if len(self.accel_history) > self.window_size:
            self.accel_history.pop(0)

        return self._analyze()

    def _analyze(self):
        if len(self.accel_history) < self.window_size:
            return "INITIALIZING", 0

        recent_data = np.array(self.accel_history[-5:])
        avg_accel = np.mean(self.accel_history)
        var_accel = np.var(recent_data)

        # Stationary: lage variance en lage gemiddelde dynamische accel
        if var_accel < self.stat_var_threshold and abs(avg_accel) < self.stat_mean_threshold:
            self.last_state = "STATIONARY"
            return self.last_state, self.step_count

        # Step: piek in recente samples
        if np.max(recent_data) > self.step_threshold and self.last_state != "STEP_DETECTED":
            self.step_count += 1
            self.last_state = "STEP_DETECTED"
            return self.last_state, self.step_count

        self.last_state = "MOVING"
        return self.last_state, self.step_count

# ========== 3. VELOCITY INTEGRATION MODULE ==========
class VelocityIntegrator:
    """Integrates acceleration (m/s^2) to estimate velocity and position."""
    def __init__(self):
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)
        self.last_time = time.time()
        self.stationary_correction = True  # ZUPT-like correction

    def update(self, linear_accel_mps2, is_stationary=False, dt=None):
        if dt is None:
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time

        # Integrate
        self.velocity += linear_accel_mps2 * dt
        self.position += self.velocity * dt

        # Stationary -> reset velocity (jouw punt 3)
        if is_stationary and self.stationary_correction:
            self.velocity = np.zeros(3)

        return self.velocity.copy(), self.position.copy(), dt

# ========== 4. MAIN PIPELINE ORCHESTRATION ==========
def main_pipeline(sample_rate_hz=50):
    print(f"[INFO] Starting pipeline at {sample_rate_hz}Hz")
    print("[INFO] Press Ctrl+C to stop and save data.")

    # 1D accel KFs (optie A)
    kf_x = Kalman1DAccelRandomWalk()
    kf_y = Kalman1DAccelRandomWalk()
    kf_z = Kalman1DAccelRandomWalk()

    step_detector = SimpleStepDetector()
    integrator = VelocityIntegrator()

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sensor_pipeline_log_{timestamp_str}.csv"

    header = [
        'timestamp', 'dt',
        'accel_raw_x_g', 'accel_raw_y_g', 'accel_raw_z_g',
        'accel_raw_x_mps2', 'accel_raw_y_mps2', 'accel_raw_z_mps2',
        'accel_filt_x_mps2', 'accel_filt_y_mps2', 'accel_filt_z_mps2',
        'gyro_x', 'gyro_y', 'gyro_z',
        'step_state', 'step_count',
        'velocity_x', 'velocity_y', 'velocity_z',
        'pos_x', 'pos_y', 'pos_z'
    ]

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        sample_count = 0
        start_time = time.time()

        while running:
            loop_start = time.time()
            sample_count += 1

            # ---- A. READ RAW SENSOR DATA ----
            accel_raw = sense.get_accelerometer_raw()  # in g
            gyro_raw = sense.get_gyroscope_raw()       # rad/s (SenseHat convention)
            timestamp = time.time()

            accel_g = np.array([accel_raw['x'], accel_raw['y'], accel_raw['z']], dtype=float)
            gyro_vector = np.array([gyro_raw['x'], gyro_raw['y'], gyro_raw['z']], dtype=float)

            # ---- B. FIX UNITS: g -> m/s^2 (jouw punt 2) ----
            accel_mps2 = accel_g * G_TO_MPS2

            # ---- C. APPLY ACCEL-KF (Optie A) ----
            accel_filt_mps2 = np.array([
                kf_x.update(accel_mps2[0]),
                kf_y.update(accel_mps2[1]),
                kf_z.update(accel_mps2[2])
            ], dtype=float)

            # ---- D. STEP / STATIONARY DETECTION (op filtered accel) ----
            step_state, step_count = step_detector.add_reading(accel_filt_mps2)
            is_stationary = (step_state == "STATIONARY")

            # ---- E. VELOCITY & POSITION INTEGRATION (met velocity reset) ----
            velocity, position, dt = integrator.update(accel_filt_mps2, is_stationary)

            # ---- F. DATA LOGGING ----
            writer.writerow([
                timestamp, dt,
                accel_g[0], accel_g[1], accel_g[2],
                accel_mps2[0], accel_mps2[1], accel_mps2[2],
                accel_filt_mps2[0], accel_filt_mps2[1], accel_filt_mps2[2],
                gyro_vector[0], gyro_vector[1], gyro_vector[2],
                step_state, step_count,
                velocity[0], velocity[1], velocity[2],
                position[0], position[1], position[2]
            ])

            # ---- G. REAL-TIME VISUAL FEEDBACK ----
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

            # ---- H. CONSOLE STATUS OUTPUT (every 2 seconds) ----
            if sample_count % max(int(sample_rate_hz * 2), 1) == 0:
                elapsed = time.time() - start_time
                print(f"[STATUS] Time: {elapsed:.1f}s | "
                      f"Steps: {step_count} | "
                      f"State: {step_state} | "
                      f"Pos: ({position[0]:.2f}, {position[1]:.2f}) | "
                      f"dt: {dt*1000:.1f}ms")

            # ---- I. RATE CONTROL ----
            processing_time = time.time() - loop_start
            target_delay = 1.0 / sample_rate_hz
            sleep_time = target_delay - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sample_count % 100 == 0:
                print(f"[WARNING] Can't maintain {sample_rate_hz}Hz. "
                      f"Processing takes {processing_time:.3f}s")

    print(f"\n[INFO] Pipeline stopped. Data saved to {filename}")
    print(f"[INFO] Processed {sample_count} samples in {time.time()-start_time:.1f} seconds")
    sense.clear()

# ========== 6. EXECUTION ==========
if __name__ == "__main__":
    main_pipeline(sample_rate_hz=50)
