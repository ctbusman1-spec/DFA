from __future__ import annotations

import time

from imu.sensors.imu_reader import wrap_angle


class LiveStrideDetector:
    def __init__(self, imu_reader, threshold_g: float, min_stride_interval_s: float, fixed_step_length_m: float):
        self.imu_reader = imu_reader
        self.threshold_g = threshold_g
        self.min_stride_interval_s = min_stride_interval_s
        self.fixed_step_length_m = fixed_step_length_m
        self.last_stride_t = 0.0
        self.prev_yaw = None

    def wait_for_step(self):
        while True:
            now = time.time()
            acc_norm = self.imu_reader.get_acceleration_norm()
            yaw = self.imu_reader.get_yaw()

            if self.prev_yaw is None:
                self.prev_yaw = yaw

            heading_change = wrap_angle(yaw - self.prev_yaw)

            if acc_norm > self.threshold_g and (now - self.last_stride_t) >= self.min_stride_interval_s:
                dt = max(now - self.last_stride_t, self.min_stride_interval_s)
                self.last_stride_t = now
                self.prev_yaw = yaw
                return {
                    "timestamp": now,
                    "dt": dt,
                    "heading_change": heading_change,
                    "step_length_m": self.fixed_step_length_m,
                    "raw_heading": yaw,
                }

            self.prev_yaw = yaw
            time.sleep(0.01)
