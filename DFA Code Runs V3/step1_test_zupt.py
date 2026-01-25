#!/usr/bin/env python3
"""
STEP 1: Test ZUPT Detector with your actual CSV format
"""
import numpy as np
import pandas as pd
from zupt_detector import ZUPTDetector

print("\n" + "="*80)
print("STEP 1: TESTING ZUPT DETECTOR")
print("="*80)

# Load CSV
csv_file = 'Real-data-set-sensor_log_4.csv'
print(f"\nLoading {csv_file}...")
data = pd.read_csv(csv_file)
print(f"✓ Loaded {len(data)} samples")
print(f"✓ Columns: {list(data.columns)[:5]}... ({len(data.columns)} total)")

# Initialize ZUPT detector
zupt = ZUPTDetector()
print("\n✓ ZUPT detector initialized")

# Test on first 100 samples
print("\nTesting ZUPT detector on first 100 samples...")
stationary_count = 0

for i in range(min(100, len(data))):
    row = data.iloc[i]
    
    # Convert from 'g' to m/s^2 (1g = 9.81 m/s^2)
    acc_x = float(row['acc_x_g']) * 9.81
    acc_y = float(row['acc_y_g']) * 9.81
    acc_z = float(row['acc_z_g']) * 9.81
    accel = np.array([acc_x, acc_y, acc_z])
    
    gyro_x = float(row['gyro_x_rads'])
    gyro_y = float(row['gyro_y_rads'])
    gyro_z = float(row['gyro_z_rads'])
    gyro = np.array([gyro_x, gyro_y, gyro_z])
    
    # Velocity not available, use zeros
    vel = np.array([0.0, 0.0, 0.0])
    
    is_stat, conf = zupt.update(accel, gyro, vel)
    if is_stat:
        stationary_count += 1

print(f"✓ Processed 100 samples")
print(f"✓ Stationary phases detected: {stationary_count}/100")

print("\n" + "="*80)
print("STEP 1 COMPLETE ✓")
print("="*80)
print("\nNext: Run step2_test_linear_kf.py\n")
