#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import warnings

warnings.filterwarnings('ignore')

class BayesianFilter:
    """
    Non-recursive Bayesian Filter with Mode-Seeking
    
    Based on:
    - Koroglu & Yilmaz (2017): "Pedestrian inertial navigation with building 
      floor plans for indoor environments via non-recursive Bayesian filtering"
    - Cheng (1995): "Mean shift, mode seeking, and clustering"
    
    Theory:
    P(x|z) = P(z|x) * P(x) / P(z)
    
    Where:
    - x: state (position, velocity)
    - z: measurement (accelerometer, gyroscope)
    - P(x|z): posterior (what we want)
    - P(z|x): likelihood (sensor model)
    - P(x): prior (motion model)
    - P(z): evidence (normalization)
    """
    
    def __init__(self, process_noise=0.01, measurement_noise=0.1, 
                 window_size=10, grid_resolution=0.1):
        """
        Initialize Bayesian Filter
        
        Parameters:
        -----------
        process_noise : float
            Prior uncertainty (process noise)
        measurement_noise : float
            Likelihood uncertainty (measurement noise)
        window_size : int
            Moving window for mode-seeking
        grid_resolution : float
            Grid resolution for particle placement (m)
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.window_size = window_size
        self.grid_resolution = grid_resolution
        
        # State: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        
        # Posterior distribution
        self.posterior_mean = np.zeros(3)
        self.posterior_cov = np.eye(3) * 0.1
        
        # Mode estimate
        self.mode_position = np.zeros(3)
        self.mode_confidence = 0.0
        
        # History for mode-seeking window
        self.measurement_history = []
        self.max_history = window_size
        
    def predict_prior(self, accel, dt=0.05):
        """
        Predict prior P(x_k | z_{1:k-1})
        
        Motion model: constant velocity + acceleration
        x_{k+1} = x_k + v_k * dt + 0.5 * a_k * dt^2
        v_{k+1} = v_k + a_k * dt
        
        Parameters:
        -----------
        accel : array (3,)
            Linear acceleration measurement
        dt : float
            Time step
        
        Returns:
        --------
        prior_mean : array (3,)
            Predicted position mean
        prior_cov : array (3,3)
            Predicted position covariance
        """
        # Update velocity
        self.velocity = self.velocity + accel * dt
        
        # Update position with motion model
        self.position = self.position + self.velocity * dt + 0.5 * accel * dt**2
        
        # Prior covariance grows due to process noise
        prior_mean = self.position.copy()
        prior_cov = self.posterior_cov + np.eye(3) * self.process_noise
        
        return prior_mean, prior_cov
    
    def likelihood(self, position, measurement, measurement_cov):
        """
        Calculate likelihood P(z|x)
        
        Gaussian likelihood model:
        P(z|x) = N(z; h(x), R)
        
        Where:
        - z: measurement
        - h(x): measurement function (identity for position)
        - R: measurement noise covariance
        
        Parameters:
        -----------
        position : array (3,)
            State position
        measurement : array (3,)
            Measurement (position estimate from sensors)
        measurement_cov : array (3,3)
            Measurement covariance
        
        Returns:
        --------
        likelihood : float
            P(z|x)
        """
        try:
            diff = measurement - position
            likelihood = multivariate_normal.pdf(diff, 
                                                mean=np.zeros(3),
                                                cov=measurement_cov)
            return likelihood
        except:
            return 1e-10
    
    def prior(self, position, prior_mean, prior_cov):
        """
        Calculate prior P(x)
        
        Gaussian prior from motion model:
        P(x) = N(x; x_pred, Q)
        
        Parameters:
        -----------
        position : array (3,)
            State position
        prior_mean : array (3,)
            Prior mean from motion model
        prior_cov : array (3,3)
            Prior covariance
        
        Returns:
        --------
        prior : float
            P(x)
        """
        try:
            diff = position - prior_mean
            prior = multivariate_normal.pdf(diff,
                                          mean=np.zeros(3),
                                          cov=prior_cov)
            return prior
        except:
            return 1e-10
    
    def mode_seeking(self, positions, weights, bandwidth=0.2):
        """
        Mean-shift mode-seeking algorithm
        
        Based on Cheng (1995): Shifts estimate towards high-density region
        
        Formula:
        m(x) = sum(w_i * K(x - x_i)) / sum(K(x - x_i))
        
        Where:
        - K: kernel function (Gaussian)
        - w_i: weight at position x_i
        - m(x): mode estimate at x
        
        Parameters:
        -----------
        positions : array (N, 3)
            Candidate positions
        weights : array (N,)
            Posterior weights
        bandwidth : float
            Kernel bandwidth
        
        Returns:
        --------
        mode_pos : array (3,)
            Mode position
        mode_conf : float
            Mode confidence (max weight)
        """
        if len(positions) == 0:
            return self.position.copy(), 0.0
        
        # Normalize weights
        weights = np.maximum(weights, 1e-10)
        weights = weights / np.sum(weights)
        
        # Initialize mode at weighted mean
        mode_pos = np.average(positions, axis=0, weights=weights)
        
        # Mean-shift iterations
        for iteration in range(20):
            mode_old = mode_pos.copy()
            
            # Gaussian kernel weights
            distances = np.linalg.norm(positions - mode_pos, axis=1)
            kernel_weights = np.exp(-0.5 * (distances / bandwidth)**2)
            
            # Weighted mean shift
            total_weight = np.sum(kernel_weights * weights)
            if total_weight > 0:
                mode_pos = np.sum((kernel_weights * weights)[:, np.newaxis] * 
                                positions, axis=0) / total_weight
            
            # Convergence check
            if np.linalg.norm(mode_pos - mode_old) < 1e-6:
                break
        
        # Confidence: max posterior weight
        mode_conf = np.max(weights)
        
        return mode_pos, mode_conf
    
    def update(self, measurement, prior_mean, prior_cov):
        """
        Bayesian update: Calculate P(x|z)
        
        Bayes rule:
        P(x|z) ∝ P(z|x) * P(x)
        
        Algorithm:
        1. Generate candidate positions around prior_mean
        2. Calculate likelihood P(z|x) for each
        3. Calculate prior P(x) for each
        4. Calculate posterior ∝ likelihood * prior
        5. Use mode-seeking to find peak
        
        Parameters:
        -----------
        measurement : array (3,)
            Measurement
        prior_mean : array (3,)
            Prior mean from motion model
        prior_cov : array (3,3)
            Prior covariance
        
        Returns:
        --------
        posterior_mean : array (3,)
            Posterior mean
        posterior_cov : array (3,3)
            Posterior covariance
        """
        measurement_cov = np.eye(3) * self.measurement_noise
        
        # Generate candidate positions (grid around prior mean)
        n_candidates = 1000
        sigma = np.sqrt(np.diag(prior_cov).mean())
        positions = prior_mean[:, np.newaxis] + sigma * np.random.randn(3, n_candidates)
        positions = positions.T
        
        # Calculate posteriors: P(x|z) ∝ P(z|x) * P(x)
        posteriors = np.zeros(n_candidates)
        for i in range(n_candidates):
            likelihood = self.likelihood(positions[i], measurement, measurement_cov)
            prior = self.prior(positions[i], prior_mean, prior_cov)
            posteriors[i] = likelihood * prior
        
        # Mode-seeking on posterior distribution
        self.mode_position, self.mode_confidence = self.mode_seeking(
            positions, posteriors, bandwidth=sigma * 0.5
        )
        
        # Update posterior estimate
        self.posterior_mean = self.mode_position.copy()
        self.posterior_cov = prior_cov.copy()
        
        return self.posterior_mean, self.posterior_cov
    
    def get_state(self):
        """Return current position and velocity estimates"""
        return self.position.copy(), self.velocity.copy()
    
    def get_uncertainty(self):
        """Return uncertainty estimates"""
        pos_std = np.sqrt(np.diag(self.posterior_cov))
        vel_std = np.ones(3) * 0.1
        return pos_std, vel_std
