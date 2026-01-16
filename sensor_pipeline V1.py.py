# sensor_pipeline.py
import time
import numpy as np
from sense_hat import SenseHat
import csv
from datetime import datetime
import signal
import sys

# ========== 1. INITIALIZATION ==========
print("[INFO] Initializing Sensor Pipeline...")
sense = SenseHat()
sense.set_imu_config(True, True, False)  # Enable gyro, accel, disable mag
sense.clear()  # Clear the LED matrix

# Global flag for graceful shutdown
running = True
def signal_handler(sig, frame):
    global running
    print('[INFO] Shutdown signal received.')
    running = False
    sense.clear()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

# ========== 2. CORE PROCESSING CLASSES ==========
class LinearKalmanFilter1D:
    """A 1D Kalman Filter for smoothing a single sensor channel."""
    def __init__(self, process_variance=1e-4, measurement_variance=0.05**2):
        self.process_var = process_variance
        self.meas_var = measurement_variance
        self.state_estimate = np.array([0., 0.])  # [position, velocity]
        self.estimate_cov = np.eye(2)  # Initial uncertainty
        self.last_time = time.time()

    def update(self, measurement):
        current_time = time.time()
        dt = max(current_time - self.last_time, 0.001)  # Prevent dt=0
        self.last_time = current_time

        # PREDICTION
        F = np.array([[1, dt], [0, 1]])  # State transition
        # Process noise increases with dt
        Q = np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]]) * self.process_var
        self.state_estimate = F @ self.state_estimate
        self.estimate_cov = F @ self.estimate_cov @ F.T + Q

        # UPDATE
        H = np.array([[1., 0.]])  # Measurement matrix
        y = measurement - H @ self.state_estimate  # Innovation
        S = H @ self.estimate_cov @ H.T + self.meas_var
        K = self.estimate_cov @ H.T / S  # Kalman Gain

        self.state_estimate = self.state_estimate + K.flatten() * y
        self.estimate_cov = (np.eye(2) - np.outer(K, H)) @ self.estimate_cov

        return self.state_estimate[0]  # Smoothed position estimate

class SimpleStepDetector:
    """Detects steps and stationary phases using accelerometer magnitude."""
    def __init__(self, window_size=15, step_threshold=0.8, stationary_threshold=0.2):
        self.window_size = window_size
        self.step_threshold = step_threshold
        self.stat_threshold = stationary_threshold
        self.accel_history = []
        self.step_count = 0
        self.last_state = "MOVING"

    def add_reading(self, accel_vector, gravity=np.array([0, 0, 1])):
        # Remove gravity component (assuming IMU is upright)
        dynamic_accel = accel_vector - gravity
        accel_magnitude = np.linalg.norm(dynamic_accel)
        self.accel_history.append(accel_magnitude)

        if len(self.accel_history) > self.window_size:
            self.accel_history.pop(0)

        return self._analyze()

    def _analyze(self):
        if len(self.accel_history) < self.window_size:
            return "INITIALIZING", 0

        recent_data = np.array(self.accel_history[-5:])  # Last 5 samples
        avg_accel = np.mean(self.accel_history)
        var_accel = np.var(recent_data)  # Variance of recent samples

        # Detect stationary phase (low variance)
        if var_accel < self.stat_threshold and abs(avg_accel) < 0.3:
            self.last_state = "STATIONARY"
        # Detect step (peak in recent samples)
        elif np.max(recent_data) > self.step_threshold and self.last_state != "STEP":
            self.step_count += 1
            self.last_state = "STEP_DETECTED"
            return self.last_state, self.step_count
        else:
            self.last_state = "MOVING"

        return self.last_state, self.step_count

# ========== 3. VELOCITY INTEGRATION MODULE ==========
class VelocityIntegrator:
    """Basic dead reckoning: integrates acceleration to estimate velocity and position."""
    def __init__(self):
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)
        self.last_time = time.time()
        self.stationary_correction = True  # Apply ZUPT-like correction

    def update(self, linear_accel, is_stationary=False, dt=None):
        if dt is None:
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time

        # INTEGRATION: v = v + a*dt, p = p + v*dt
        self.velocity += linear_accel * dt
        self.position += self.velocity * dt

        # ZERO-VELOCITY UPDATE (ZUPT) Correction
        if is_stationary and self.stationary_correction:
            self.velocity = np.zeros(3)  # Reset velocity during stationary phase

        return self.velocity.copy(), self.position.copy()

# ========== 4. MAIN PIPELINE ORCHESTRATION ==========
def main_pipeline(sample_rate_hz=50):
    """Main function that runs the real-time sensor processing pipeline."""
    print(f"[INFO] Starting pipeline at {sample_rate_hz}Hz")
    print("[INFO] Press Ctrl+C to stop and save data.")

    # Initialize processing modules
    kf_x = LinearKalmanFilter1D()
    kf_y = LinearKalmanFilter1D()
    kf_z = LinearKalmanFilter1D()
    step_detector = SimpleStepDetector()
    integrator = VelocityIntegrator()

    # Data logging setup
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sensor_pipeline_log_{timestamp_str}.csv"
    header = ['timestamp', 'accel_raw_x', 'accel_raw_y', 'accel_raw_z',
              'accel_filt_x', 'accel_filt_y', 'accel_filt_z',
              'gyro_x', 'gyro_y', 'gyro_z',
              'step_state', 'step_count',
              'velocity_x', 'velocity_y', 'velocity_z',
              'pos_x', 'pos_y', 'pos_z']

    # Main processing loop
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        sample_count = 0
        start_time = time.time()

        while running:
            loop_start = time.time()
            sample_count += 1

            # ---- A. READ RAW SENSOR DATA ----
            accel_raw = sense.get_accelerometer_raw()
            gyro_raw = sense.get_gyroscope_raw()
            timestamp = time.time()

            accel_vector = np.array([accel_raw['x'], accel_raw['y'], accel_raw['z']])
            gyro_vector = np.array([gyro_raw['x'], gyro_raw['y'], gyro_raw['z']])

            # ---- B. APPLY KALMAN FILTERING ----
            accel_filtered = np.array([
                kf_x.update(accel_vector[0]),
                kf_y.update(accel_vector[1]),
                kf_z.update(accel_vector[2])
            ])

            # ---- C. STEP DETECTION ----
            step_state, step_count = step_detector.add_reading(accel_filtered)

            # ---- D. VELOCITY & POSITION INTEGRATION ----
            is_stationary = (step_state == "STATIONARY")
            velocity, position = integrator.update(accel_filtered, is_stationary)

            # ---- E. DATA LOGGING ----
            writer.writerow([
                timestamp,
                accel_vector[0], accel_vector[1], accel_vector[2],
                accel_filtered[0], accel_filtered[1], accel_filtered[2],
                gyro_vector[0], gyro_vector[1], gyro_vector[2],
                step_state, step_count,
                velocity[0], velocity[1], velocity[2],
                position[0], position[1], position[2]
            ])

            # ---- F. REAL-TIME VISUAL FEEDBACK ----
            # Show step count on LED matrix (first 64 steps only)
            if step_state == "STEP_DETECTED":
                sense.clear()
                if step_count <= 64:
                    # Light up an LED for each step
                    idx = step_count - 1
                    row = idx // 8
                    col = idx % 8
                    sense.set_pixel(col, row, (0, 100, 0))  # Green dot
            elif step_state == "STATIONARY":
                # Blink center pixel red when stationary
                if int(time.time() * 2) % 2 == 0:
                    sense.set_pixel(3, 3, (100, 0, 0))
                else:
                    sense.set_pixel(3, 3, (0, 0, 0))

            # ---- G. CONSOLE STATUS OUTPUT (every 2 seconds) ----
            if sample_count % (sample_rate_hz * 2) == 0:
                elapsed = time.time() - start_time
                print(f"[STATUS] Time: {elapsed:.1f}s | "
                      f"Steps: {step_count} | "
                      f"State: {step_state} | "
                      f"Pos: ({position[0]:.2f}, {position[1]:.2f})")

            # ---- H. RATE CONTROL ----
            processing_time = time.time() - loop_start
            target_delay = 1.0 / sample_rate_hz
            sleep_time = target_delay - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sample_count % 100 == 0:
                print(f"[WARNING] Can't maintain {sample_rate_hz}Hz. "
                      f"Processing takes {processing_time:.3f}s")

    # Shutdown cleanup
    print(f"\n[INFO] Pipeline stopped. Data saved to {filename}")
    print(f"[INFO] Processed {sample_count} samples in {time.time()-start_time:.1f} seconds")
    sense.clear()

# ========== 5. POST-PROCESSING ANALYSIS TOOL ==========
def analyze_log_file(filename):
    """Quick analysis of the logged data - to be used in your Jupyter notebook."""
    import pandas as pd
    import matplotlib.pyplot as plt
    
    print(f"[ANALYSIS] Loading {filename}")
    df = pd.read_csv(filename)
    
    # Create a simple plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 1. Raw vs Filtered Acceleration
    axes[0].plot(df['timestamp'] - df['timestamp'].iloc[0], 
                 df['accel_raw_x'], 'gray', alpha=0.5, label='Raw X')
    axes[0].plot(df['timestamp'] - df['timestamp'].iloc[0], 
                 df['accel_filt_x'], 'blue', label='Filtered X')
    axes[0].set_ylabel('Acceleration (G)')
    axes[0].legend()
    axes[0].set_title('Kalman Filter Performance')
    
    # 2. Step Detection
    step_indices = df[df['step_state'] == 'STEP_DETECTED'].index
    axes[1].plot(df['timestamp'] - df['timestamp'].iloc[0], 
                 df['accel_filt_y'], 'green')
    axes[1].scatter(df['timestamp'].iloc[step_indices] - df['timestamp'].iloc[0],
                   df['accel_filt_y'].iloc[step_indices], color='red', marker='o')
    axes[1].set_ylabel('Accel Y (G)')
    axes[1].set_title(f'Step Detection ({len(step_indices)} steps found)')
    
    # 3. Integrated Position (Dead Reckoning)
    axes[2].plot(df['pos_x'], df['pos_y'], 'b-')
    axes[2].plot(df['pos_x'].iloc[0], df['pos_y'].iloc[0], 'go', markersize=10, label='Start')
    axes[2].plot(df['pos_x'].iloc[-1], df['pos_y'].iloc[-1], 'ro', markersize=10, label='End')
    axes[2].set_xlabel('X Position (m)')
    axes[2].set_ylabel('Y Position (m)')
    axes[2].set_title('Dead Reckoning Trajectory')
    axes[2].legend()
    axes[2].axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'{filename}_analysis.png', dpi=150)
    plt.show()
    
    # Print statistics
    print(f"\n[STATISTICS]")
    print(f"Total samples: {len(df)}")
    print(f"Total steps detected: {df['step_count'].max()}")
    print(f"Final position: ({df['pos_x'].iloc[-1]:.2f}, {df['pos_y'].iloc[-1]:.2f})")
    print(f"Maximum velocity: {np.sqrt(df['velocity_x']**2 + df['velocity_y']**2).max():.2f} m/s")
    
    return df

# ========== 6. EXECUTION ==========
if __name__ == "__main__":
    # Run the main pipeline at 50Hz
    main_pipeline(sample_rate_hz=50)
    
    # To analyze a saved log file (uncomment and specify filename):
    # analyze_log_file('sensor_pipeline_log_20240320_143022.csv')