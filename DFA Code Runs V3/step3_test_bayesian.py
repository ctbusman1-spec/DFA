#!/usr/bin/env python3
"""
STEP 3: Test Bayesian Filter - CORRECTED (Complete Version)
Uses Linear KF estimate as measurement input
"""
import numpy as np
import pandas as pd
from zupt_detector import ZUPTDetector
from linear_kalman_filter import LinearKalmanFilter
from bayesian_filter import BayesianFilter
import matplotlib.pyplot as plt

print("\n" + "="*80)
print("STEP 3: TESTING BAYESIAN FILTER (CORRECTED)")
print("="*80)

csv_file = 'Real-data-set-sensor_log_4.csv'
print(f"\nLoading {csv_file}...")
data = pd.read_csv(csv_file)
print(f"✓ Loaded {len(data)} samples")

linear_kf = LinearKalmanFilter()
linear_kf.initialize_state([0, 0, 0], [0, 0, 0])
bayesian = BayesianFilter(process_noise=0.02, measurement_noise=0.15)
zupt = ZUPTDetector()

print("✓ Bayesian Filter initialized")

print(f"\nProcessing {len(data)} samples through Bayesian Filter...")
positions_linear = []
positions_bayesian = []
errors_linear = []
errors_bayesian = []

for i in range(len(data)):
    if (i + 1) % 200 == 0:
        print(f"  Processed {i + 1}/{len(data)}")
    
    row = data.iloc[i]
    
    accel = np.array([float(row['acc_x_g']) * 9.81,
                      float(row['acc_y_g']) * 9.81,
                      float(row['acc_z_g']) * 9.81])
    
    gyro = np.array([float(row['gyro_x_rads']), 
                     float(row['gyro_y_rads']), 
                     float(row['gyro_z_rads'])])
    
    vel = np.array([0.0, 0.0, 0.0])
    is_stat, conf = zupt.update(accel, gyro, vel)
    
    # LINEAR KALMAN FILTER
    linear_kf.predict()
    if np.max(np.abs(accel)) > 0.1:
        linear_kf.update_accelerometer(accel)
    if is_stat:
        linear_kf.update_zupt()
    pos_linear, _ = linear_kf.get_state()
    positions_linear.append(pos_linear)
    
    # BAYESIAN FILTER - CORRECTED: Use Linear KF output as measurement
    prior_mean, prior_cov = bayesian.predict_prior(accel)
    measurement = pos_linear.copy()  # KEY FIX: Use Linear KF, not ground truth
    pos_bayesian, _ = bayesian.update(measurement, prior_mean, np.eye(3) * 0.05)
    positions_bayesian.append(pos_bayesian)
    
    # Ground truth
    gt_pos = np.array([float(row['pos_x_m']), float(row['pos_y_m']), 0.0])
    errors_linear.append(np.linalg.norm(pos_linear - gt_pos))
    errors_bayesian.append(np.linalg.norm(pos_bayesian - gt_pos))

positions_linear = np.array(positions_linear)
positions_bayesian = np.array(positions_bayesian)
errors_linear = np.array(errors_linear)
errors_bayesian = np.array(errors_bayesian)

mae_linear = np.mean(errors_linear)
rmse_linear = np.sqrt(np.mean(errors_linear**2))
mae_bayesian = np.mean(errors_bayesian)
rmse_bayesian = np.sqrt(np.mean(errors_bayesian**2))

improvement = (mae_linear - mae_bayesian) / mae_linear * 100

print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)

print("\nLinear Kalman Filter:")
print(f"  MAE:   {mae_linear:.4f} m")
print(f"  RMSE:  {rmse_linear:.4f} m")

print("\nBayesian Filter:")
print(f"  MAE:   {mae_bayesian:.4f} m")
print(f"  RMSE:  {rmse_bayesian:.4f} m")
print(f"  → Improvement: {improvement:+.1f}%")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.plot(positions_linear[:, 0], positions_linear[:, 1], 'b-', label='Linear KF', linewidth=1.5)
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_title('Linear Kalman Filter: 2D Trajectory')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')

ax = axes[0, 1]
ax.plot(positions_bayesian[:, 0], positions_bayesian[:, 1], 'r-', label='Bayesian', linewidth=1.5)
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_title('Bayesian Filter: 2D Trajectory')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')

ax = axes[1, 0]
ax.plot(errors_linear, 'b-', label='Linear KF', alpha=0.7)
ax.axhline(y=mae_linear, color='b', linestyle='--', label=f'MAE={mae_linear:.4f}')
ax.set_xlabel('Sample')
ax.set_ylabel('Error (m)')
ax.set_title('Linear Kalman Filter: Error')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(errors_bayesian, 'r-', label='Bayesian', alpha=0.7)
ax.axhline(y=mae_bayesian, color='r', linestyle='--', label=f'MAE={mae_bayesian:.4f}')
ax.set_xlabel('Sample')
ax.set_ylabel('Error (m)')
ax.set_title('Bayesian Filter: Error')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step3_bayesian_comparison.png', dpi=100)
print("\n✓ Saved: step3_bayesian_comparison.png")

print("\n" + "="*80)
print("STEP 3 COMPLETE ✓")
print("="*80)
print("\nNext: Run step4_test_particle.py\n")