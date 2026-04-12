from __future__ import annotations

"""
Simplified stationary detector for the DFA IMU assignment.

Important note:
This is a practical surrogate for classical foot-mounted ZUPT. The course brief
explicitly allows replacing full pedestrian ZUPT with a simpler detection
mechanism. This implementation detects likely stationary periods using a short
window over acceleration, gyroscope activity, and step quietness.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Optional, Tuple

import numpy as np

from config import get_config


@dataclass(frozen=True)
class DetectorStatistics:
    """Snapshot of internal detector statistics."""

    is_stationary: bool
    confidence: float
    accel_mean_norm_mps2: float
    accel_variance_mps2: float
    gyro_mean_norm_rads: float
    gyro_variance_rads: float
    recent_step_count: int
    samples_processed: int


class ZUPTDetector:
    """
    Simplified stationary detector.

    The detector is intentionally lightweight and explainable:
    - acceleration magnitude should stay near gravity
    - acceleration variance should be small
    - gyroscope magnitude should be small and stable
    - recent step events should be absent

    This is suitable for the resit assignment because it is transparent,
    inexpensive, and easy to validate on Raspberry Pi hardware.
    """

    def __init__(
        self,
        window_size: Optional[int] = None,
        accel_variance_threshold: Optional[float] = None,
        gyro_variance_threshold: Optional[float] = None,
        confirmation_samples: Optional[int] = None,
        gravity_mps2: Optional[float] = None,
        accel_magnitude_tolerance_mps2: float = 1.25,
        gyro_magnitude_threshold_rads: float = 0.35,
        use_step_quietness: bool = True,
    ) -> None:
        cfg = get_config()
        self.window_size = window_size or cfg.zupt.window_size
        self.accel_variance_threshold = (
            accel_variance_threshold
            if accel_variance_threshold is not None
            else cfg.zupt.accel_variance_threshold
        )
        self.gyro_variance_threshold = (
            gyro_variance_threshold
            if gyro_variance_threshold is not None
            else cfg.zupt.gyro_variance_threshold
        )
        self.confirmation_samples = confirmation_samples or cfg.zupt.confirmation_samples
        self.gravity_mps2 = gravity_mps2 or cfg.units.gravity_mps2
        self.accel_magnitude_tolerance_mps2 = accel_magnitude_tolerance_mps2
        self.gyro_magnitude_threshold_rads = gyro_magnitude_threshold_rads
        self.use_step_quietness = use_step_quietness

        self.accel_norm_buffer: Deque[float] = deque(maxlen=self.window_size)
        self.gyro_norm_buffer: Deque[float] = deque(maxlen=self.window_size)
        self.step_buffer: Deque[int] = deque(maxlen=self.window_size)
        self.decision_buffer: Deque[bool] = deque(maxlen=self.confirmation_samples)

        self.is_stationary = False
        self.confidence = 0.0
        self.accel_mean_norm_mps2 = 0.0
        self.accel_variance_mps2 = 0.0
        self.gyro_mean_norm_rads = 0.0
        self.gyro_variance_rads = 0.0
        self.recent_step_count = 0
        self.samples_processed = 0

    def update(
        self,
        accel_mps2: Iterable[float],
        gyro_rads: Iterable[float],
        step_event: int | bool = 0,
    ) -> Tuple[bool, float]:
        """
        Update detector with one sample.

        Parameters
        ----------
        accel_mps2:
            3D acceleration vector in m/s².
        gyro_rads:
            3D gyroscope vector in rad/s.
        step_event:
            Optional binary step indicator from the dataset.

        Returns
        -------
        is_stationary, confidence
        """
        accel = np.asarray(accel_mps2, dtype=float).reshape(3)
        gyro = np.asarray(gyro_rads, dtype=float).reshape(3)
        step_flag = int(bool(step_event))

        accel_norm = float(np.linalg.norm(accel))
        gyro_norm = float(np.linalg.norm(gyro))

        self.accel_norm_buffer.append(accel_norm)
        self.gyro_norm_buffer.append(gyro_norm)
        self.step_buffer.append(step_flag)
        self.samples_processed += 1

        self._update_statistics()
        decision = self._instantaneous_decision()
        self.decision_buffer.append(decision)

        if len(self.decision_buffer) == self.confirmation_samples:
            stationary_votes = sum(self.decision_buffer)
            self.is_stationary = stationary_votes == self.confirmation_samples
            self.confidence = stationary_votes / self.confirmation_samples
        else:
            self.is_stationary = False
            self.confidence = sum(self.decision_buffer) / max(1, self.confirmation_samples)

        return self.is_stationary, float(self.confidence)

    def _update_statistics(self) -> None:
        accel_arr = np.asarray(self.accel_norm_buffer, dtype=float)
        gyro_arr = np.asarray(self.gyro_norm_buffer, dtype=float)
        step_arr = np.asarray(self.step_buffer, dtype=int)

        self.accel_mean_norm_mps2 = float(accel_arr.mean()) if accel_arr.size else 0.0
        self.accel_variance_mps2 = float(accel_arr.var()) if accel_arr.size else 0.0
        self.gyro_mean_norm_rads = float(gyro_arr.mean()) if gyro_arr.size else 0.0
        self.gyro_variance_rads = float(gyro_arr.var()) if gyro_arr.size else 0.0
        self.recent_step_count = int(step_arr.sum()) if step_arr.size else 0

    def _instantaneous_decision(self) -> bool:
        if len(self.accel_norm_buffer) < self.window_size:
            return False

        criterion_accel_level = (
            abs(self.accel_mean_norm_mps2 - self.gravity_mps2)
            <= self.accel_magnitude_tolerance_mps2
        )
        criterion_accel_variance = self.accel_variance_mps2 <= self.accel_variance_threshold
        criterion_gyro_level = self.gyro_mean_norm_rads <= self.gyro_magnitude_threshold_rads
        criterion_gyro_variance = self.gyro_variance_rads <= self.gyro_variance_threshold
        criterion_step_quiet = (self.recent_step_count == 0) if self.use_step_quietness else True

        criteria = [
            criterion_accel_level,
            criterion_accel_variance,
            criterion_gyro_level,
            criterion_gyro_variance,
            criterion_step_quiet,
        ]
        return bool(all(criteria))

    def get_statistics(self) -> Dict[str, float | int | bool]:
        """Return current detector statistics as a dictionary."""
        stats = DetectorStatistics(
            is_stationary=self.is_stationary,
            confidence=self.confidence,
            accel_mean_norm_mps2=self.accel_mean_norm_mps2,
            accel_variance_mps2=self.accel_variance_mps2,
            gyro_mean_norm_rads=self.gyro_mean_norm_rads,
            gyro_variance_rads=self.gyro_variance_rads,
            recent_step_count=self.recent_step_count,
            samples_processed=self.samples_processed,
        )
        return stats.__dict__.copy()

    def reset(self) -> None:
        """Reset detector state."""
        self.accel_norm_buffer.clear()
        self.gyro_norm_buffer.clear()
        self.step_buffer.clear()
        self.decision_buffer.clear()
        self.is_stationary = False
        self.confidence = 0.0
        self.accel_mean_norm_mps2 = 0.0
        self.accel_variance_mps2 = 0.0
        self.gyro_mean_norm_rads = 0.0
        self.gyro_variance_rads = 0.0
        self.recent_step_count = 0
        self.samples_processed = 0


if __name__ == "__main__":
    from data_loader import DataLoader

    loader = DataLoader()
    df, info = loader.load_default_dataset()

    detector = ZUPTDetector()
    stationary_flags = []
    confidences = []

    for row in df.itertuples(index=False):
        accel = [row.acc_x_mps2, row.acc_y_mps2, row.acc_z_mps2]
        gyro = [row.gyro_x_rads, row.gyro_y_rads, row.gyro_z_rads]
        is_stat, conf = detector.update(accel, gyro, row.step_event)
        stationary_flags.append(is_stat)
        confidences.append(conf)

    print("Loaded dataset:", info.source_name)
    print("Dataset type:", info.dataset_type)
    print("Samples:", len(df))
    print("Stationary detections:", int(np.sum(stationary_flags)))
    print("Mean confidence:", round(float(np.mean(confidences)), 4))
    print("Final detector statistics:")
    for key, value in detector.get_statistics().items():
        print(f"- {key}: {value}")
