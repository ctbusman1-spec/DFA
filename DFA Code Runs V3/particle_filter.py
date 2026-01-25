#!/usr/bin/env python3
import numpy as np
from scipy.stats import multivariate_normal
import warnings

warnings.filterwarnings('ignore')

class ParticleFilter:
    """Particle Filter for Pedestrian Inertial Navigation"""
    
    def __init__(self, n_particles=1000, process_noise=0.01, 
                 measurement_noise=0.1, resampling_threshold=0.5):
        self.n_particles = n_particles
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.resampling_threshold = resampling_threshold
        
        self.particles = np.random.randn(n_particles, 3) * 0.01
        self.velocities = np.zeros((n_particles, 3))
        self.weights = np.ones(n_particles) / n_particles
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.resampling_count = 0
        self.effective_sample_size = n_particles
        
    def predict(self, accel, dt=0.05):
        """Prediction step with constant acceleration model"""
        self.velocities = self.velocities + accel[np.newaxis, :] * dt
        velocity_noise = np.random.randn(self.n_particles, 3) * self.process_noise
        self.velocities = self.velocities + velocity_noise
        
        accel_term = 0.5 * accel[np.newaxis, :] * (dt ** 2)
        self.particles = self.particles + self.velocities * dt + accel_term
        
        position_noise = np.random.randn(self.n_particles, 3) * (self.process_noise * dt)
        self.particles = self.particles + position_noise
    
    def update(self, measurement, measurement_cov=None):
        """Update step: Weight particles by measurement likelihood"""
        if measurement_cov is None:
            measurement_cov = np.eye(3) * (self.measurement_noise ** 2)
        
        likelihoods = np.zeros(self.n_particles)
        for i in range(self.n_particles):
            diff = measurement - self.particles[i]
            try:
                likelihoods[i] = multivariate_normal.pdf(
                    diff, mean=np.zeros(3), cov=measurement_cov
                )
            except:
                likelihoods[i] = 1e-10
        
        self.weights = likelihoods * self.weights
        weight_sum = np.sum(self.weights)
        
        if weight_sum > 0:
            self.weights = self.weights / weight_sum
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles
        
        self.effective_sample_size = 1.0 / np.sum(self.weights ** 2)
        
        if self.effective_sample_size < self.resampling_threshold * self.n_particles:
            self.resample()
    
    def resample(self, method='multinomial'):
        """Resampling step: duplicate high-weight particles"""
        if method == 'multinomial':
            indices = np.random.choice(self.n_particles, 
                                      size=self.n_particles,
                                      p=self.weights)
        else:
            indices = np.random.choice(self.n_particles,
                                      size=self.n_particles,
                                      p=self.weights)
        
        self.particles = self.particles[indices]
        self.velocities = self.velocities[indices]
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.resampling_count += 1
    
    def estimate_state(self, method='weighted_mean'):
        """Estimate state from particles"""
        if method == 'weighted_mean':
            self.position = np.average(self.particles, 
                                      axis=0, 
                                      weights=self.weights)
            self.velocity = np.average(self.velocities,
                                      axis=0,
                                      weights=self.weights)
        
        return self.position.copy(), self.velocity.copy()
    
    def get_state(self):
        """Return current position and velocity estimates"""
        return self.position.copy(), self.velocity.copy()
