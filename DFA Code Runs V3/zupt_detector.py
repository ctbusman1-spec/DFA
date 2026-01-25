"""
ZUPT (Zero Velocity Update) Detector
=====================================

Implements stationary phase detection using multiple criteria:
1. Accelerometer magnitude variance
2. Gyroscope magnitude variance  
3. Velocity thresholding
4. Step state confirmation

Optimized for Raspberry Pi - O(n) complexity, minimal memory
Reference: Jiménez et al. (2010) Indoor Navigation System Using IMU

Author: IMU Assignment Part 2
Date: 2026-01-25
"""

import numpy as np
from collections import deque
import warnings


class ZUPTDetector:
    """
    Zero Velocity Update Detector
    
    Detects stationary periods from IMU measurements using adaptive thresholding.
    
    Parameters
    ----------
    window_size : int
        Number of samples for moving window statistics (default: 10)
    accel_threshold : float
        Variance threshold for accelerometer (m/s²)² (default: 0.01)
    gyro_threshold : float
        Variance threshold for gyroscope (rad/s)² (default: 1e-5)
    velocity_threshold : float
        Velocity magnitude threshold (m/s) (default: 0.05)
    confirmation_samples : int
        Samples needed to confirm stationary state (default: 5)
    """
    
    def __init__(self, 
                 window_size=10,
                 accel_threshold=0.01,
                 gyro_threshold=1e-5,
                 velocity_threshold=0.05,
                 confirmation_samples=5):
        
        self.window_size = window_size
        self.accel_threshold = accel_threshold
        self.gyro_threshold = gyro_threshold
        self.velocity_threshold = velocity_threshold
        self.confirmation_samples = confirmation_samples
        
        self.is_stationary = False
        self.zero_velocity_vector = np.zeros(3)
        
        self.accel_buffer = deque(maxlen=window_size)
        self.gyro_buffer = deque(maxlen=window_size)
        self.velocity_buffer = deque(maxlen=window_size)
        self.detection_history = deque(maxlen=confirmation_samples)
        
        self.accel_variance = 0.0
        self.gyro_variance = 0.0
        self.velocity_magnitude = 0.0
        
        self.samples_processed = 0
    
    def update(self, accel, gyro, velocity=None, dt=0.05):
        """
        Update ZUPT detector with new IMU measurements.
        
        Parameters
        ----------
        accel : array_like
            Accelerometer measurement [ax, ay, az] in m/s²
        gyro : array_like
            Gyroscope measurement [wx, wy, wz] in rad/s
        velocity : array_like, optional
            Velocity measurement [vx, vy, vz] in m/s
        dt : float, optional
            Time since last update (seconds)
        
        Returns
        -------
        is_stationary : bool
            True if current measurements indicate stationary state
        confidence : float
            Confidence score from 0.0 to 1.0
        """
        
        accel = np.asarray(accel, dtype=np.float32)
        gyro = np.asarray(gyro, dtype=np.float32)
        
        self.accel_buffer.append(accel)
        self.gyro_buffer.append(gyro)
        if velocity is not None:
            velocity = np.asarray(velocity, dtype=np.float32)
            self.velocity_buffer.append(velocity)
            self.velocity_magnitude = np.linalg.norm(velocity)
        
        self.samples_processed += 1
        
        if len(self.accel_buffer) == self.window_size:
            accel_array = np.array(self.accel_buffer)
            gyro_array = np.array(self.gyro_buffer)
            
            self.accel_variance = np.var(accel_array)
            self.gyro_variance = np.var(gyro_array)
        
        criterion_accel = self.accel_variance < self.accel_threshold
        criterion_gyro = self.gyro_variance < self.gyro_threshold
        criterion_velocity = self.velocity_magnitude < self.velocity_threshold
        
        detection = criterion_accel and criterion_gyro and criterion_velocity
        self.detection_history.append(detection)
        
        if len(self.detection_history) == self.confirmation_samples:
            self.is_stationary = all(self.detection_history)
        else:
            self.is_stationary = False
        
        n_criteria = 3
        n_met = sum([criterion_accel, criterion_gyro, criterion_velocity])
        confidence = n_met / n_criteria * len(self.detection_history) / self.confirmation_samples
        confidence = min(1.0, confidence)
        
        return self.is_stationary, confidence
    
    def get_statistics(self):
        """Get current detection statistics."""
        return {
            'is_stationary': self.is_stationary,
            'accel_variance': self.accel_variance,
            'gyro_variance': self.gyro_variance,
            'velocity_magnitude': self.velocity_magnitude,
            'samples_processed': self.samples_processed,
        }
    
    def reset(self):
        """Reset detector to initial state."""
        self.accel_buffer.clear()
        self.gyro_buffer.clear()
        self.velocity_buffer.clear()
        self.detection_history.clear()
        self.is_stationary = False
        self.accel_variance = 0.0
        self.gyro_variance = 0.0
        self.velocity_magnitude = 0.0
        self.samples_processed = 0
