#!/usr/bin/env python3
"""
STEP 2: Test Linear Kalman Filter with actual CSV format
"""
import numpy as np
import pandas as pd
from zupt_detector import ZUPTDetector
from linear_kalman_filter import LinearKalmanFilter
import matplotlib.pyplot as plt

print("\n" + "="*80)
print("STEP 2: TESTING LINEAR KALMAN FILTER")
print("="*80)

# Load CSV
csv_file = 'Real-data-set-sensor_log_4.csv'
print(f"\nLoading {csv_file}...")
data = pd.read_csv(csv_file)
print(f"✓ Loaded {len(data)} samples")

# Initialize filters
linear_kf = LinearKalmanFilter()
linear_kf.initialize_state([0, 0, 0], [0, 0, 0])
zupt = ZUPTDetector()

print("✓ Linear Kalman Filter initialized")

# Test on all samples
print(f"\nProcessing {len(data)} samples through Linear KF...")
positions = []
errors = []

for i in range(len(data)):
    if (i + 1) % 200 == 0:
        print(f"  Processed {i + 1}/{len(data)}")
    
    row = data.iloc[i]
    
    # Convert accelerations from 'g' to m/s^2
    acc_x = float(row['acc_x_g']) * 9.81
    acc_y = float(row['acc_y_g']) * 9.81
    acc_z = float(row['acc_z_g']) * 9.81
    accel = np.array([acc_x, acc_y, acc_z])
    
    # Gyro data
    gyro_x = float(row['gyro_x_rads'])
    gyro_y = float(row['gyro_y_rads'])
    gyro_z = float(row['gyro_z_rads'])
    gyro = np.array([gyro_x, gyro_y, gyro_z])
    
    vel = np.array([0.0, 0.0, 0.0])
    
    # ZUPT detection
    is_stat, conf = zupt.update(accel, gyro, vel)
    
    # Linear KF prediction and update
    linear_kf.predict()
    if np.max(np.abs(accel)) > 0.1:
        linear_kf.update_accelerometer(accel)
    if is_stat:
        linear_kf.update_zupt()
    
    pos, vel_est = linear_kf.get_state()
    positions.append(pos)
    
    # Ground truth position
    gt_pos = np.array([float(row['pos_x_m']), float(row['pos_y_m']), 0.0])
    error = np.linalg.norm(pos - gt_pos)
    errors.append(error)

positions = np.array(positions)
errors = np.array(errors)

# Results
mae = np.mean(errors)
rmse = np.sqrt(np.mean(errors**2))
max_error = np.max(errors)

print("\n" + "="*80)
print("LINEAR KALMAN FILTER RESULTS")
print("="*80)
print(f"\nMAE (Mean Absolute Error):  {mae:.4f} m")
print(f"RMSE (Root Mean Sq Error):  {rmse:.4f} m")
print(f"Max Error:                   {max_error:.4f} m")
print(f"\nPosition range (X): {positions[:, 0].min():.2f} to {positions[:, 0].max():.2f} m")
print(f"Position range (Y): {positions[:, 1].min():.2f} to {positions[:, 1].max():.2f} m")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(positions[:, 0], positions[:, 1], 'b-', label='Estimated Path')
ax1.set_xlabel('X Position (m)')
ax1.set_ylabel('Y Position (m)')
ax1.set_title('Linear Kalman Filter: 2D Trajectory')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

ax2.plot(errors, 'b-', linewidth=1)
ax2.axhline(y=mae, color='r', linestyle='--', label=f'MAE = {mae:.4f} m')
ax2.set_xlabel('Sample')
ax2.set_ylabel('Position Error (m)')
ax2.set_title('Linear Kalman Filter: Error Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step2_linear_kf_results.png', dpi=100)
print("\n✓ Saved: step2_linear_kf_results.png")

print("\n" + "="*80)
print("STEP 2 COMPLETE ✓")
print("="*80)
print("\nNext: Run step3_test_bayesian.py\n")
