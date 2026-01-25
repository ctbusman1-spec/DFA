#!/usr/bin/env python3
"""
STEP 5: All Filters Comparison - COMPLETE CORRECTED VERSION
Compares Linear KF, Bayesian, and Particle Filters
All filters use proper measurement inputs (Linear KF output)
"""
import numpy as np
import pandas as pd
from zupt_detector import ZUPTDetector
from linear_kalman_filter import LinearKalmanFilter
from bayesian_filter import BayesianFilter
from particle_filter import ParticleFilter
import matplotlib.pyplot as plt
import os

print("\n" + "="*80)
print("STEP 5: ALL FILTERS COMPARISON (COMPLETE)")
print("="*80)

csv_file = 'Real-data-set-sensor_log_4.csv'
print(f"\nLoading {csv_file}...")
data = pd.read_csv(csv_file)
print(f"✓ Loaded {len(data)} samples")

linear_kf = LinearKalmanFilter()
linear_kf.initialize_state([0, 0, 0], [0, 0, 0])
bayesian = BayesianFilter(process_noise=0.02, measurement_noise=0.15)
particle = ParticleFilter(n_particles=500, process_noise=0.015, 
                         measurement_noise=0.25, resampling_threshold=0.8)
zupt = ZUPTDetector()

print("✓ All filters initialized")

print(f"\nProcessing {len(data)} samples through all filters...")
time_array = data['t'].values
positions_linear = []
positions_bayesian = []
positions_particle = []
ground_truth = []
errors_linear = []
errors_bayesian = []
errors_particle = []

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
    
    gt_pos = np.array([float(row['pos_x_m']), float(row['pos_y_m']), 0.0])
    ground_truth.append(gt_pos)
    
    # ===== LINEAR KALMAN FILTER =====
    linear_kf.predict()
    if np.max(np.abs(accel)) > 0.1:
        linear_kf.update_accelerometer(accel)
    if is_stat:
        linear_kf.update_zupt()
    pos_linear, _ = linear_kf.get_state()
    positions_linear.append(pos_linear)
    errors_linear.append(np.linalg.norm(pos_linear - gt_pos))
    
    # ===== BAYESIAN FILTER - CORRECTED =====
    prior_mean, prior_cov = bayesian.predict_prior(accel)
    measurement_bayesian = pos_linear.copy()  # KEY FIX: Use Linear KF output
    pos_bayesian, _ = bayesian.update(measurement_bayesian, prior_mean, np.eye(3) * 0.05)
    positions_bayesian.append(pos_bayesian)
    errors_bayesian.append(np.linalg.norm(pos_bayesian - gt_pos))
    
    # ===== PARTICLE FILTER - CORRECTED =====
    particle.predict(accel)
    particle.update(pos_linear, measurement_cov)  # KEY FIX: Use Linear KF output
    pos_particle, _ = particle.estimate_state(method='weighted_mean')
    positions_particle.append(pos_particle)
    errors_particle.append(np.linalg.norm(pos_particle - gt_pos))

positions_linear = np.array(positions_linear)
positions_bayesian = np.array(positions_bayesian)
positions_particle = np.array(positions_particle)
ground_truth = np.array(ground_truth)
errors_linear = np.array(errors_linear)
errors_bayesian = np.array(errors_bayesian)
errors_particle = np.array(errors_particle)

mae_linear = np.mean(errors_linear)
rmse_linear = np.sqrt(np.mean(errors_linear**2))
max_linear = np.max(errors_linear)

mae_bayesian = np.mean(errors_bayesian)
rmse_bayesian = np.sqrt(np.mean(errors_bayesian**2))
max_bayesian = np.max(errors_bayesian)

mae_particle = np.mean(errors_particle)
rmse_particle = np.sqrt(np.mean(errors_particle**2))
max_particle = np.max(errors_particle)

print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

print("\n1. LINEAR KALMAN FILTER (Baseline)")
print("-" * 80)
print(f"   MAE:       {mae_linear:.4f} m")
print(f"   RMSE:      {rmse_linear:.4f} m")
print(f"   Max Error: {max_linear:.4f} m")

print("\n2. BAYESIAN FILTER (Mode-Seeking)")
print("-" * 80)
print(f"   MAE:       {mae_bayesian:.4f} m")
print(f"   RMSE:      {rmse_bayesian:.4f} m")
print(f"   Max Error: {max_bayesian:.4f} m")
improvement_b = (mae_linear - mae_bayesian) / mae_linear * 100
print(f"   Improvement vs Linear: {improvement_b:+.1f}%")

print("\n3. PARTICLE FILTER")
print("-" * 80)
print(f"   MAE:       {mae_particle:.4f} m")
print(f"   RMSE:      {rmse_particle:.4f} m")
print(f"   Max Error: {max_particle:.4f} m")
improvement_p = (mae_linear - mae_particle) / mae_linear * 100
print(f"   Improvement vs Linear: {improvement_p:+.1f}%")

print("\n" + "="*80)
print("COMPARISON TABLE")
print("="*80)
print(f"{'Method':<25} {'MAE (m)':<12} {'RMSE (m)':<12} {'Max (m)':<12}")
print("-" * 62)
print(f"{'Linear Kalman':<25} {mae_linear:<12.4f} {rmse_linear:<12.4f} {max_linear:<12.4f}")
print(f"{'Bayesian':<25} {mae_bayesian:<12.4f} {rmse_bayesian:<12.4f} {max_bayesian:<12.4f}")
print(f"{'Particle':<25} {mae_particle:<12.4f} {rmse_particle:<12.4f} {max_particle:<12.4f}")

print("\n" + "="*80)
print("CREATING VISUALIZATIONS...")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

ax = axes[0, 0]
ax.plot(time_array, ground_truth[:, 0], 'g-', linewidth=2, label='Ground Truth')
ax.plot(time_array, positions_linear[:, 0], 'b--', linewidth=1.5, label='Linear KF', alpha=0.8)
ax.plot(time_array, positions_bayesian[:, 0], 'r--', linewidth=1.5, label='Bayesian', alpha=0.8)
ax.plot(time_array, positions_particle[:, 0], 'orange', linestyle='--', linewidth=1.5, label='Particle', alpha=0.8)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position X (m)')
ax.set_title('Position X: All Methods')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(time_array, ground_truth[:, 1], 'g-', linewidth=2, label='Ground Truth')
ax.plot(time_array, positions_linear[:, 1], 'b--', linewidth=1.5, label='Linear KF', alpha=0.8)
ax.plot(time_array, positions_bayesian[:, 1], 'r--', linewidth=1.5, label='Bayesian', alpha=0.8)
ax.plot(time_array, positions_particle[:, 1], 'orange', linestyle='--', linewidth=1.5, label='Particle', alpha=0.8)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position Y (m)')
ax.set_title('Position Y: All Methods')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
ax.plot(ground_truth[:, 0], ground_truth[:, 1], 'g-', linewidth=2.5, label='Ground Truth', marker='o', markersize=2, markevery=100)
ax.plot(positions_linear[:, 0], positions_linear[:, 1], 'b--', linewidth=1.5, label='Linear KF', alpha=0.7)
ax.plot(positions_bayesian[:, 0], positions_bayesian[:, 1], 'r--', linewidth=1.5, label='Bayesian', alpha=0.7)
ax.plot(positions_particle[:, 0], positions_particle[:, 1], 'orange', linestyle='--', linewidth=1.5, label='Particle', alpha=0.7)
ax.plot(ground_truth[0, 0], ground_truth[0, 1], 'go', markersize=12, label='Start')
ax.plot(ground_truth[-1, 0], ground_truth[-1, 1], 'r*', markersize=18, label='End')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_title('2D Trajectory Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')

ax = axes[1, 0]
ax.plot(time_array, errors_linear, 'b-', linewidth=1.5, alpha=0.8)
ax.axhline(y=mae_linear, color='b', linestyle=':', linewidth=2, label=f'MAE={mae_linear:.4f}m')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Error (m)')
ax.set_title('Linear Kalman Filter: Error')
ax.legend()
ax.grid(True, alpha=0.3)
ax.fill_between(time_array, 0, errors_linear, alpha=0.2)

ax = axes[1, 1]
ax.plot(time_array, errors_bayesian, 'r-', linewidth=1.5, alpha=0.8)
ax.axhline(y=mae_bayesian, color='r', linestyle=':', linewidth=2, label=f'MAE={mae_bayesian:.4f}m')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Error (m)')
ax.set_title('Bayesian Filter: Error')
ax.legend()
ax.grid(True, alpha=0.3)
ax.fill_between(time_array, 0, errors_bayesian, alpha=0.2, color='red')

ax = axes[1, 2]
ax.plot(time_array, errors_particle, 'orange', linewidth=1.5, alpha=0.8)
ax.axhline(y=mae_particle, color='orange', linestyle=':', linewidth=2, label=f'MAE={mae_particle:.4f}m')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Error (m)')
ax.set_title('Particle Filter: Error')
ax.legend()
ax.grid(True, alpha=0.3)
ax.fill_between(time_array, 0, errors_particle, alpha=0.2, color='orange')

plt.tight_layout()
plt.savefig('step5_all_filters_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: step5_all_filters_comparison.png")

os.makedirs('results', exist_ok=True)
results_df = pd.DataFrame({
    'time': time_array,
    'gt_x': ground_truth[:, 0],
    'gt_y': ground_truth[:, 1],
    'linear_x': positions_linear[:, 0],
    'linear_y': positions_linear[:, 1],
    'bayesian_x': positions_bayesian[:, 0],
    'bayesian_y': positions_bayesian[:, 1],
    'particle_x': positions_particle[:, 0],
    'particle_y': positions_particle[:, 1],
    'error_linear': errors_linear,
    'error_bayesian': errors_bayesian,
    'error_particle': errors_particle,
})

results_df.to_csv('results/all_filters_results.csv', index=False)
print("✓ Saved: results/all_filters_results.csv")

print("\n" + "="*80)
print("STEP 5 COMPLETE ✓")
print("="*80)
print("\nNext: Run python3 validate_floorplan.py\n")