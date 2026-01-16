import numpy as np

class SimpleStepDetector:
    # This class is used to roughly detect steps using accelerometer data.
    # It is a simpler alternative to a full ZUPT algorithm.

    def __init__(self, window_size=10, accel_threshold=0.5, stationary_threshold=0.2):
        self.window_size = window_size
        self.accel_threshold = accel_threshold
        self.stationary_threshold = stationary_threshold

        # Store recent acceleration magnitudes
        self.accel_history = []

    def add_accel_reading(self, accel_vector):
        # Estimate acceleration magnitude
        # Subtract gravity (around 1G on Z axis)
        gravity = np.array([0, 0, 1])
        corrected_accel = accel_vector - gravity

        # Calculate magnitude manually step by step
        magnitude = np.linalg.norm(corrected_accel)

        # Save value in history
        self.accel_history.append(magnitude)

        # Keep history size fixed
        if len(self.accel_history) > self.window_size:
            self.accel_history.pop(0)

        # Analyze current movement
        return self.analyze_state()

    def analyze_state(self):
        # Not enough data yet
        if len(self.accel_history) < self.window_size:
            return "INSUFFICIENT_DATA"

        # Calculate average and variance
        avg_accel = np.mean(self.accel_history)
        accel_variance = np.var(self.accel_history)

        # If acceleration changes very little, assume foot is stationary
        if accel_variance < self.stationary_threshold and abs(avg_accel) < 0.3:
            return "STATIONARY"

        # If acceleration is large enough, assume a step happened
        if avg_accel > self.accel_threshold:
            return "STEP_DETECTED"

        # Otherwise, the person is moving but no clear step
        return "MOVING"


# Example usage
detector = SimpleStepDetector()

# Inside main loop (example):
# accel_data = read_imu_data()['accel']
# state = detector.add_accel_reading(accel_data)
#
# if state == "STATIONARY":
#     print("Foot is stationary, velocity can be reset")
