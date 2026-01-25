#!/usr/bin/env python3
"""
STEP 4: Test Particle Filter - CORRECTED (Complete Version)
Uses Linear KF estimate as measurement input
"""
import numpy as np
import pandas as pd
from zupt_detector import ZUPTDetector
from linear_kalman_filter import LinearKalmanFilter
from particle_filter import ParticleFilter
import matplotlib.pyplot as plt

print("\n" + "="*80)
print("STEP 4: TESTING PARTICLE FILTER (CORRECTED)")
print("="*80)

csv_file = 'Real-data-set-sensor_log_4.csv'
print(f"\nLoading {csv_file}...")
data = pd.read_csv(csv_file)
print(f"✓ Loaded {len(data)} samples")

linear_kf = LinearKalmanFilter()
linear_kf.initialize_state([0, 0, 0], [0, 0, 0])
particle = ParticleFilter(n_particles=500, process_noise=0.015, 
                         measurement_noise=0.25, resampling_threshold=0.8)
zupt = ZUPTDetector()

print("✓ Particle Filter initialized (500 particles)")

print(f"\nProcessing {len(data)} samples through Particle Filter...")
positions_linear = []
positions_particle = []
errors_linear = []
errors_particle = []
resampling_events = []

measurement_cov = np.eye(3) * 0.1

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
    
    # PARTICLE FILTER - CORRECTED: Use Linear KF output as measurement
    particle.predict(accel)
    measurement = pos_linear.copy()  # KEY FIX: Use Linear KF, not ground truth
    particle.update(measurement, measurement_cov)
    pos_particle, _ = particle.estimate_state(method='weighted_mean')
    positions_particle.append(pos_particle)
    resampling_events.append(particle.resampling_count)
    
    # Ground truth
    gt_pos = np.array([float(row['pos_x_m']), float(row['pos_y_m']), 0.0])
    errors_linear.append(np.linalg.norm(pos_linear - gt_pos))
    errors_particle.append(np.linalg.norm(pos_particle - gt_pos))

positions_linear = np.array(positions_linear)
positions_particle = np.array(positions_particle)
errors_linear = np.array(errors_linear)
errors_particle = np.array(errors_particle)

mae_linear = np.mean(errors_linear)
rmse_linear = np.sqrt(np.mean(errors_linear**2))
mae_particle = np.mean(errors_particle)
rmse_particle = np.sqrt(np.mean(errors_particle**2))

improvement = (mae_linear - mae_particle) / mae_linear * 100
total_resamplings = resampling_events[-1] if resampling_events else 0

print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)

print("\nLinear Kalman Filter:")
print(f"  MAE:   {mae_linear:.4f} m")
print(f"  RMSE:  {rmse_linear:.4f} m")

print("\nParticle Filter:")
print(f"  MAE:   {mae_particle:.4f} m")
print(f"  RMSE:  {rmse_particle:.4f} m")
print(f"  → Improvement: {improvement:+.1f}%")
print(f"  → Total resamplings: {total_resamplings}")

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
ax.plot(positions_particle[:, 0], positions_particle[:, 1], 'orange', label='Particle Filter', linewidth=1.5)
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_title('Particle Filter: 2D Trajectory')
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
ax.plot(errors_particle, 'orange', label='Particle', alpha=0.7)
ax.axhline(y=mae_particle, color='orange', linestyle='--', label=f'MAE={mae_particle:.4f}')
ax.set_xlabel('Sample')
ax.set_ylabel('Error (m)')
ax.set_title('Particle Filter: Error')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step4_particle_comparison.png', dpi=100)
print("\n✓ Saved: step4_particle_comparison.png")

print("\n" + "="*80)
print("STEP 4 COMPLETE ✓")
print("="*80)
print("\nNext: Run step5_all_filters_comparison.py\n")