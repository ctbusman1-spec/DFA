from __future__ import annotations

"""
Batch experiment runner for the DFA IMU assignment.

This script runs the full pipeline on:
- the labeled benchmark dataset for quantitative accuracy,
- the extra Raspberry Pi logs for robustness and generalization analysis.

It saves one combined summary table so the notebook can compare both groups
without pretending the unlabeled Pi logs have ground-truth error metrics.
"""

from pathlib import Path
import json
import time
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from config import get_config
from run_experiment import ExperimentRunner


class BatchExperimentRunner:
    def __init__(self) -> None:
        self.cfg = get_config()
        self.runner = ExperimentRunner()
        self.results_dir = self.cfg.paths.results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def discover_datasets(self) -> list[Path]:
        names = [
            "Real-data-set-sensor_log_4.csv",
            "pi_sensor_log_walk_straight.csv",
            "pi_sensor_log_walk_slight_turn.csv",
            "pi_sensor_log_walk_turn.csv",
            "pi_sensor_log_turnwalk.csv",
            "pi_sensor_log_turn90.csv",
            "pi_sensor_log_20260325_135712.csv",
        ]
        search_roots = [
            self.cfg.paths.data_dir,
            self.cfg.paths.project_root,
            Path.cwd(),
            Path("/mnt/data"),
        ]
        found: list[Path] = []
        seen = set()
        for name in names:
            for root in search_roots:
                candidate = root / name
                if candidate.exists():
                    resolved = str(candidate.resolve())
                    if resolved not in seen:
                        found.append(candidate)
                        seen.add(resolved)
                    break
        return found

    @staticmethod
    def _path_length(xs: Iterable[float], ys: Iterable[float]) -> float:
        xs = np.asarray(list(xs), dtype=float)
        ys = np.asarray(list(ys), dtype=float)
        if len(xs) < 2:
            return 0.0
        dx = np.diff(xs)
        dy = np.diff(ys)
        return float(np.sum(np.sqrt(dx**2 + dy**2)))

    @staticmethod
    def _final_displacement(xs: Iterable[float], ys: Iterable[float]) -> float:
        xs = np.asarray(list(xs), dtype=float)
        ys = np.asarray(list(ys), dtype=float)
        if len(xs) == 0:
            return 0.0
        return float(np.sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2))

    @staticmethod
    def _trajectory_smoothness(xs: Iterable[float], ys: Iterable[float]) -> float:
        xs = np.asarray(list(xs), dtype=float)
        ys = np.asarray(list(ys), dtype=float)
        if len(xs) < 3:
            return 0.0
        vx = np.diff(xs)
        vy = np.diff(ys)
        ax = np.diff(vx)
        ay = np.diff(vy)
        jerk_like = np.sqrt(ax**2 + ay**2)
        return float(np.mean(jerk_like))

    @staticmethod
    def _heading_range_rad(source_df: pd.DataFrame) -> float:
        heading = source_df["heading_rad"].dropna().to_numpy(dtype=float)
        if len(heading) == 0:
            return 0.0
        return float(np.max(heading) - np.min(heading))

    def _motion_label(self, dataset_name: str) -> str:
        name = dataset_name.lower()
        if "straight" in name:
            return "straight_walk"
        if "slight_turn" in name:
            return "slight_turn"
        if "turn90" in name:
            return "turn_90"
        if "turnwalk" in name:
            return "turn_then_walk"
        if "walk_turn" in name:
            return "walk_with_turn"
        if "sensor_log_4" in name:
            return "benchmark_route"
        return "other"

    def run_one(self, dataset_path: Path) -> dict:
        source_df, source_info = self.runner.loader.load_csv(dataset_path)

        t0 = time.perf_counter()
        result = self.runner.run(dataset_path)
        runtime_s = time.perf_counter() - t0

        df = result["trajectories"]
        metrics = result["metrics"]
        ds = metrics["dataset"]

        row = {
            "dataset_name": ds["source_name"],
            "dataset_type": ds["dataset_type"],
            "motion_label": self._motion_label(ds["source_name"]),
            "n_samples": ds["n_samples"],
            "has_ground_truth": ds["has_ground_truth"],
            "has_gyro": ds["has_gyro"],
            "duration_s": round(float(source_df["time_s"].iloc[-1] - source_df["time_s"].iloc[0]), 6) if len(source_df) >= 2 else 0.0,
            "mean_dt_s": round(float(source_df["dt_s"].mean()), 8),
            "step_events": int(source_df["step_event"].sum()),
            "zupt_stationary_count": int(df["zupt_stationary"].sum()),
            "zupt_stationary_rate": round(float(df["zupt_stationary"].mean()), 6),
            "zupt_mean_confidence": round(float(df["zupt_confidence"].mean()), 6),
            "heading_range_rad": round(self._heading_range_rad(source_df), 6),
            "runtime_s": round(float(runtime_s), 6),
            "kf_path_length_m": round(self._path_length(df["kf_pos_x_m"], df["kf_pos_y_m"]), 6),
            "bayes_path_length_m": round(self._path_length(df["bayes_pos_x_m"], df["bayes_pos_y_m"]), 6),
            "particle_path_length_m": round(self._path_length(df["particle_pos_x_m"], df["particle_pos_y_m"]), 6),
            "kf_final_displacement_m": round(self._final_displacement(df["kf_pos_x_m"], df["kf_pos_y_m"]), 6),
            "bayes_final_displacement_m": round(self._final_displacement(df["bayes_pos_x_m"], df["bayes_pos_y_m"]), 6),
            "particle_final_displacement_m": round(self._final_displacement(df["particle_pos_x_m"], df["particle_pos_y_m"]), 6),
            "kf_smoothness": round(self._trajectory_smoothness(df["kf_pos_x_m"], df["kf_pos_y_m"]), 8),
            "bayes_smoothness": round(self._trajectory_smoothness(df["bayes_pos_x_m"], df["bayes_pos_y_m"]), 8),
            "particle_smoothness": round(self._trajectory_smoothness(df["particle_pos_x_m"], df["particle_pos_y_m"]), 8),
            "bayes_mode_confidence": round(float(metrics["bayesian"]["mode_confidence"]), 8),
            "particle_resampling_count": int(metrics["particle"]["resampling_count"]),
            "linear_mae_2d_m": metrics["linear_kalman_filter"]["mae_2d_m"],
            "linear_rmse_2d_m": metrics["linear_kalman_filter"]["rmse_2d_m"],
            "bayes_mae_2d_m": metrics["bayesian_filter"]["mae_2d_m"],
            "bayes_rmse_2d_m": metrics["bayesian_filter"]["rmse_2d_m"],
            "particle_mae_2d_m": metrics["particle_filter"]["mae_2d_m"],
            "particle_rmse_2d_m": metrics["particle_filter"]["rmse_2d_m"],
            "floorplan_walkability_percent": None,
        }

        if metrics.get("floorplan_walkability") is not None:
            row["floorplan_walkability_percent"] = metrics["floorplan_walkability"]["walkability_percent"]

        if row["linear_mae_2d_m"] is not None:
            baseline = float(row["linear_mae_2d_m"])
            row["bayes_mae_change_vs_linear_pct"] = round((float(row["bayes_mae_2d_m"]) - baseline) / max(baseline, 1e-9) * 100.0, 6)
            row["particle_mae_change_vs_linear_pct"] = round((float(row["particle_mae_2d_m"]) - baseline) / max(baseline, 1e-9) * 100.0, 6)
        else:
            row["bayes_mae_change_vs_linear_pct"] = None
            row["particle_mae_change_vs_linear_pct"] = None

        return row

    def run_all(self, dataset_paths: Optional[list[Path]] = None) -> dict:
        paths = self.discover_datasets() if dataset_paths is None else dataset_paths
        if not paths:
            raise FileNotFoundError("No datasets found for batch processing.")

        rows = []
        for path in paths:
            print("=" * 80)
            print(f"Running dataset: {path.name}")
            row = self.run_one(path)
            rows.append(row)
            print(f"- dataset_type: {row['dataset_type']}")
            print(f"- runtime_s: {row['runtime_s']}")
            print(f"- step_events: {row['step_events']}")
            print(f"- zupt_stationary_count: {row['zupt_stationary_count']}")
            if row['linear_mae_2d_m'] is not None:
                print(f"- linear_mae_2d_m: {row['linear_mae_2d_m']}")
                print(f"- bayes_mae_2d_m: {row['bayes_mae_2d_m']}")
                print(f"- particle_mae_2d_m: {row['particle_mae_2d_m']}")
            else:
                print(f"- kf_path_length_m: {row['kf_path_length_m']}")
                print(f"- bayes_path_length_m: {row['bayes_path_length_m']}")
                print(f"- particle_path_length_m: {row['particle_path_length_m']}")

        summary_df = pd.DataFrame(rows)
        summary_csv = self.results_dir / "batch_experiment_summary.csv"
        summary_json = self.results_dir / "batch_experiment_summary.json"
        summary_df.to_csv(summary_csv, index=False)
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

        return {
            "summary_df": summary_df,
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
        }


if __name__ == "__main__":
    runner = BatchExperimentRunner()
    result = runner.run_all()
    print("=" * 80)
    print("BATCH SUMMARY SAVED")
    print("=" * 80)
    print("CSV:", result["summary_csv"])
    print("JSON:", result["summary_json"])
    print()
    print(result["summary_df"].to_string(index=False))
