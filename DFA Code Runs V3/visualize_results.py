#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zupt_detector import ZUPTDetector
from linear_kalman_filter import LinearKalmanFilter

print("Running Complete Test with Visualization")
print("="*70)

csv_file = 'Real-data-set-sensor_log_4.csv'

print("Loading:", csv_file)
data = pd.read_csv(csv_file)
print("Loaded", len(data), "samples")

zupt = ZUPTDetector()
kf = LinearKalmanFilter()

try:
    pos0 = data['pos_x'].iloc[0]
    kf.initialize_state([pos0, data['pos_y'].iloc[0], data['pos_z'].iloc[0]], [0,0,0])
except:
    kf.initialize_state([0, 0, 0], [0, 0, 0])

time = (data['timestamp'].values - data['timestamp'].values[0])

positions_est = []
positions_gt = []
stationary = []

print("Processing data...")

for i in range(len(data)):
    row = data.iloc[i]
    
    ax = float(row['accel_raw_x_mps2'])
    ay = float(row['accel_raw_y_mps2'])
    az = float(row['accel_raw_z_mps2'])
    
    gx = float(row['gyro_x'])
    gy = float(row['gyro_y'])
    gz = float(row['gyro_z'])
    
    vx = float(row['velocity_x'])
    vy = float(row['velocity_y'])
    vz = float(row['velocity_z'])
    
    accel = np.array([ax, ay, az])
    gyro = np.array([gx, gy, gz])
    vel = np.array([vx, vy, vz])
    
    is_stat, conf = zupt.update(accel, gyro, vel)
    
    kf.predict()
    
    alin_x = float(row['accel_lin_x_mps2'])
    alin_y = float(row['accel_lin_y_mps2'])
    alin_z = float(row['accel_lin_z_mps2'])
    
    alin = np.array([alin_x, alin_y, alin_z])
    if np.max(np.abs(alin)) > 0.001:
        kf.update_accelerometer(alin)
    
    if is_stat:
        kf.update_zupt()
    
    pos, vel_est = kf.get_state()
    positions_est.append(pos)
    
    gt_pos = np.array([float(row['pos_x']), float(row['pos_y']), float(row['pos_z'])])
    positions_gt.append(gt_pos)
    
    stationary.append(is_stat)

positions_est = np.array(positions_est)
positions_gt = np.array(positions_gt)
stationary = np.array(stationary)

print("Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.plot(time, positions_gt[:, 0], 'g-', linewidth=2, label='Ground Truth X')
ax.plot(time, positions_est[:, 0], 'b--', linewidth=1.5, label='Estimated X')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position X (m)')
ax.set_title('Position X Component')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(time, positions_gt[:, 1], 'g-', linewidth=2, label='Ground Truth Y')
ax.plot(time, positions_est[:, 1], 'b--', linewidth=1.5, label='Estimated Y')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position Y (m)')
ax.set_title('Position Y Component')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(positions_gt[:, 0], positions_gt[:, 1], 'g-', linewidth=2, label='Ground Truth')
ax.plot(positions_est[:, 0], positions_est[:, 1], 'b--', linewidth=1.5, label='Estimated')
ax.plot(positions_gt[0, 0], positions_gt[0, 1], 'go', markersize=10, label='Start')
ax.plot(positions_gt[-1, 0], positions_gt[-1, 1], 'r*', markersize=15, label='End')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_title('2D Trajectory (Top View)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')

errors = np.linalg.norm(positions_est - positions_gt, axis=1)
ax = axes[1, 1]
ax.plot(time, errors, 'r-', linewidth=1.5, label='Position Error')
ax.fill_between(time, 0, errors, alpha=0.3)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Error (m)')
ax.set_title('Position Error Over Time')
ax.grid(True, alpha=0.3)
mae = np.mean(errors)
ax.axhline(y=mae, color='k', linestyle='--', label=f'MAE={mae:.4f}m')
ax.legend()

plt.tight_layout()
plt.savefig('kalman_filter_results.png', dpi=150, bbox_inches='tight')
print("Saved visualization to: kalman_filter_results.png")

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print("Samples processed:", len(data))
print("Stationary detected:", np.sum(stationary), "/", len(data))
print("MAE:", round(np.mean(errors), 4), "m")
print("RMSE:", round(np.sqrt(np.mean(errors**2)), 4), "m")
print("Max Error:", round(np.max(errors), 4), "m")
print("="*70)
