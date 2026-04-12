class LinearKalmanFilter1D:
    """
    A simple 1D Kalman filter for smoothing sensor data (e.g., one axis of the accelerometer).
    """
    def __init__(self, process_variance=1e-3, measurement_variance=0.1 ** 2):
        # Process and measurement noise
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        # State: [position, velocity]
        self.state_estimate = np.array([0., 0.])  # Initial state (value, derivative)
        self.estimate_covariance = np.eye(2)  # Initial uncertainty

    def update(self, measurement, dt=0.01):
        """
        Performs a prediction and update step.
        :param measurement: New sensor reading.
        :param dt: Time step since last update.
        """
        # PREDICTION STEP (State Transition: x = x + v*dt; v = v)
        F = np.array([[1, dt],
                      [0, 1]])
        # Process noise matrix
        Q = np.array([[dt**4/4, dt**3/2],
                      [dt**3/2, dt**2]]) * self.process_variance

        self.state_estimate = F @ self.state_estimate
        self.estimate_covariance = F @ self.estimate_covariance @ F.T + Q

        # UPDATE STEP
        H = np.array([[1., 0.]])  # We only measure position (the value)
        y = measurement - H @ self.state_estimate  # Innovation
        S = H @ self.estimate_covariance @ H.T + self.measurement_variance  # Innovation covariance
        K = self.estimate_covariance @ H.T / S  # Kalman Gain

        self.state_estimate = self.state_estimate + K * y
        self.estimate_covariance = (np.eye(2) - K @ H) @ self.estimate_covariance

        return self.state_estimate[0]  # Return the smoothed estimate

# Example: Smoothing the x-axis acceleration
kf = LinearKalmanFilter1D()
smoothed_accel_x = kf.update(current_accel_x_reading, dt=0.01)