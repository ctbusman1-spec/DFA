from __future__ import annotations

"""
Run the full DFA IMU experiment pipeline.

This script connects:
- normalized data loading,
- simplified stationary detection,
- linear Kalman filter baseline,
- Bayesian mode-seeking filter,
- particle filter,
- static floorplan prior,
- metric and trajectory export.

Outputs are written into the configured outputs/results folder so the notebook can
reuse them directly.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional
import json

import numpy as np
import pandas as pd

from config import get_config
from data_loader import DataLoader
from zupt_detector import ZUPTDetector
from linear_kalman_filter import LinearKalmanFilter
from bayesian_filter import BayesianFilter
from particle_filter import ParticleFilter
from floorplan_utils import FloorplanBuilder, FloorplanValidator


@dataclass(frozen=True)
class MethodMetrics:
    mae_2d_m: Optional[float]
    rmse_2d_m: Optional[float]
    max_error_2d_m: Optional[float]


class ExperimentRunner:
    def __init__(self) -> None:
        self.cfg = get_config()
        self.loader = DataLoader()
        self.floorplan_builder = FloorplanBuilder()
        self.floorplan_builder.build_prior()
        self.floorplan_validator = FloorplanValidator(self.floorplan_builder)

    @staticmethod
    def _compute_metrics(errors: list[float]) -> MethodMetrics:
        if not errors:
            return MethodMetrics(None, None, None)
        arr = np.asarray(errors, dtype=float)
        return MethodMetrics(
            mae_2d_m=float(arr.mean()),
            rmse_2d_m=float(np.sqrt(np.mean(arr**2))),
            max_error_2d_m=float(arr.max()),
        )

    def run(self, dataset_path: Optional[str | Path] = None) -> dict:
        dataset = self.cfg.paths.data_file if dataset_path is None else Path(dataset_path)
        df, info = self.loader.load_csv(dataset)

        kf = LinearKalmanFilter(planar_only=True)
        bayes = BayesianFilter(candidate_count=400)
        pf = ParticleFilter(n_particles=self.cfg.particle.n_particles, planar_only=True)
        detector = ZUPTDetector()

        bayes.set_static_prior(self.floorplan_builder.prior_probability_at_meters)

        if info.has_ground_truth:
            init_pos = [float(df['gt_pos_x_m'].iloc[0]), float(df['gt_pos_y_m'].iloc[0]), 0.0]
        else:
            init_pos = [0.0, 0.0, 0.0]

        kf.initialize_state(init_pos, [0.0, 0.0, 0.0])
        bayes.initialize_state(init_pos, [0.0, 0.0, 0.0])
        pf.initialize_state(init_pos, [0.0, 0.0, 0.0])

        records: list[dict] = []
        errors_kf: list[float] = []
        errors_bayes: list[float] = []
        errors_pf: list[float] = []

        last_step_xy = np.array(init_pos[:2], dtype=float)
        stride_length_m: Optional[float] = None

        for row in df.itertuples(index=False):
            accel = np.array([row.acc_x_mps2, row.acc_y_mps2, row.acc_z_mps2], dtype=float)
            gyro = np.array([row.gyro_x_rads, row.gyro_y_rads, row.gyro_z_rads], dtype=float)

            # Linear KF baseline.
            kf.predict(row.dt_s)
            if np.max(np.abs(accel[:2])) > self.cfg.evaluation.zero_accel_update_threshold_mps2:
                kf.update_accelerometer(accel)

            is_stationary, zupt_conf = detector.update(accel, gyro, row.step_event)
            if is_stationary:
                kf.update_zero_velocity()
            kf.update_floor_constraint()
            pos_kf, vel_kf = kf.get_state()

            # Derive approximate stride length from KF only when a step event occurs.
            if int(row.step_event) > 0:
                stride_length_m = float(np.linalg.norm(pos_kf[:2] - last_step_xy))
                last_step_xy = pos_kf[:2].copy()
                if not np.isfinite(stride_length_m) or stride_length_m <= 0:
                    stride_length_m = None

            # Bayesian filter uses KF estimate as measurement, never ground truth.
            prior_mean, prior_cov = bayes.predict_prior(accel, row.dt_s)
            meas_cov_bayes = np.eye(3, dtype=float) * 0.05
            pos_bayes, cov_bayes = bayes.update(
                measurement=pos_kf,
                prior_mean=prior_mean,
                prior_cov=prior_cov,
                measurement_cov=meas_cov_bayes,
                stride_length_m=stride_length_m,
            )

            # Particle filter uses KF estimate as measurement, never ground truth.
            pf.predict(accel, row.dt_s)
            meas_cov_pf = np.eye(3, dtype=float) * 0.03
            pf.update(pos_kf, meas_cov_pf)
            pos_pf, vel_pf = pf.estimate_state(method="weighted_mean")

            gt_x = float(row.gt_pos_x_m) if np.isfinite(row.gt_pos_x_m) else np.nan
            gt_y = float(row.gt_pos_y_m) if np.isfinite(row.gt_pos_y_m) else np.nan

            err_kf = err_bayes = err_pf = np.nan
            if info.has_ground_truth and np.isfinite(gt_x) and np.isfinite(gt_y):
                gt_xy = np.array([gt_x, gt_y], dtype=float)
                err_kf = float(np.linalg.norm(pos_kf[:2] - gt_xy))
                err_bayes = float(np.linalg.norm(pos_bayes[:2] - gt_xy))
                err_pf = float(np.linalg.norm(pos_pf[:2] - gt_xy))
                errors_kf.append(err_kf)
                errors_bayes.append(err_bayes)
                errors_pf.append(err_pf)

            records.append(
                {
                    "sample_idx": int(row.sample_idx),
                    "time_s": float(row.time_s),
                    "dt_s": float(row.dt_s),
                    "step_event": int(row.step_event),
                    "zupt_stationary": bool(is_stationary),
                    "zupt_confidence": float(zupt_conf),
                    "gt_pos_x_m": gt_x,
                    "gt_pos_y_m": gt_y,
                    "kf_pos_x_m": float(pos_kf[0]),
                    "kf_pos_y_m": float(pos_kf[1]),
                    "kf_pos_z_m": float(pos_kf[2]),
                    "kf_vel_x_mps": float(vel_kf[0]),
                    "kf_vel_y_mps": float(vel_kf[1]),
                    "bayes_pos_x_m": float(pos_bayes[0]),
                    "bayes_pos_y_m": float(pos_bayes[1]),
                    "bayes_pos_z_m": float(pos_bayes[2]),
                    "particle_pos_x_m": float(pos_pf[0]),
                    "particle_pos_y_m": float(pos_pf[1]),
                    "particle_pos_z_m": float(pos_pf[2]),
                    "kf_error_2d_m": err_kf,
                    "bayes_error_2d_m": err_bayes,
                    "particle_error_2d_m": err_pf,
                }
            )

        trajectories_df = pd.DataFrame.from_records(records)
        metrics = {
            "dataset": {
                "source_name": info.source_name,
                "dataset_type": info.dataset_type,
                "n_samples": info.n_samples,
                "has_ground_truth": info.has_ground_truth,
                "has_gyro": info.has_gyro,
            },
            "linear_kalman_filter": self._metrics_to_dict(self._compute_metrics(errors_kf)),
            "bayesian_filter": self._metrics_to_dict(self._compute_metrics(errors_bayes)),
            "particle_filter": self._metrics_to_dict(self._compute_metrics(errors_pf)),
            "zupt": detector.get_statistics(),
            "kalman": kf.get_statistics(),
            "bayesian": bayes.get_statistics(),
            "particle": pf.get_statistics(),
        }

        if info.has_ground_truth:
            valid, walkability = self.floorplan_validator.validate_xy_trajectory(df['gt_pos_x_m'], df['gt_pos_y_m'])
            metrics["floorplan_walkability"] = self.floorplan_validator.report_dict(walkability)
        else:
            metrics["floorplan_walkability"] = None

        self._save_outputs(Path(dataset), trajectories_df, metrics)
        return {"trajectories": trajectories_df, "metrics": metrics}

    @staticmethod
    def _metrics_to_dict(m: MethodMetrics) -> Dict[str, Optional[float]]:
        return {
            "mae_2d_m": None if m.mae_2d_m is None else round(m.mae_2d_m, 6),
            "rmse_2d_m": None if m.rmse_2d_m is None else round(m.rmse_2d_m, 6),
            "max_error_2d_m": None if m.max_error_2d_m is None else round(m.max_error_2d_m, 6),
        }

    def _save_outputs(self, dataset: Path, trajectories_df: pd.DataFrame, metrics: dict) -> None:
        stem = dataset.stem
        results_dir = self.cfg.paths.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        traj_path = results_dir / f"{stem}_trajectories.csv"
        metrics_path = results_dir / f"{stem}_metrics.json"

        trajectories_df.to_csv(traj_path, index=False)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    def print_summary(self, metrics: dict) -> None:
        print("=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        ds = metrics["dataset"]
        print(f"Dataset: {ds['source_name']}")
        print(f"Type: {ds['dataset_type']}")
        print(f"Samples: {ds['n_samples']}")
        print(f"Ground truth: {ds['has_ground_truth']}")
        print()
        for key in ["linear_kalman_filter", "bayesian_filter", "particle_filter"]:
            vals = metrics[key]
            print(f"{key}:")
            print(f"  MAE  = {vals['mae_2d_m']}")
            print(f"  RMSE = {vals['rmse_2d_m']}")
            print(f"  MAX  = {vals['max_error_2d_m']}")
        print()
        if metrics.get("floorplan_walkability") is not None:
            fw = metrics["floorplan_walkability"]
            print("floorplan_walkability:")
            print(f"  inside_walkable = {fw['inside_walkable']}/{fw['total_samples']}")
            print(f"  walkability_percent = {fw['walkability_percent']}")
        print()
        print(f"Saved trajectories to: {self.cfg.paths.results_dir}")


if __name__ == "__main__":
    runner = ExperimentRunner()
    result = runner.run()
    runner.print_summary(result["metrics"])
