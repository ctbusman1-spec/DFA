
from __future__ import annotations

"""
Evaluation and visualization script for the DFA IMU assignment.

This script reads the saved outputs from run_experiment.py and generates:
- trajectory comparison figure,
- position-over-time figure,
- error comparison figure,
- summary CSV for quick notebook tables.

The plots are intentionally simple and report-friendly so they can be reused
inside the final notebook and PDF with minimal editing.
"""

from pathlib import Path
import json
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import get_config


class ResultsEvaluator:
    def __init__(self) -> None:
        self.cfg = get_config()
        self.results_dir = self.cfg.paths.results_dir
        self.figures_dir = self.cfg.paths.figures_dir
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_input_files(self, stem: Optional[str] = None) -> tuple[Path, Path]:
        if stem is None:
            stem = self.cfg.paths.data_file.stem
        traj_path = self.results_dir / f"{stem}_trajectories.csv"
        metrics_path = self.results_dir / f"{stem}_metrics.json"
        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
        return traj_path, metrics_path

    @staticmethod
    def _load_metrics(metrics_path: Path) -> dict:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _build_summary_table(metrics: dict) -> pd.DataFrame:
        rows = []
        for method_key, label in [
            ("linear_kalman_filter", "Linear Kalman Filter"),
            ("bayesian_filter", "Bayesian Filter"),
            ("particle_filter", "Particle Filter"),
        ]:
            vals = metrics[method_key]
            rows.append(
                {
                    "method": label,
                    "mae_2d_m": vals["mae_2d_m"],
                    "rmse_2d_m": vals["rmse_2d_m"],
                    "max_error_2d_m": vals["max_error_2d_m"],
                }
            )
        df = pd.DataFrame(rows)
        baseline_mae = float(df.loc[df["method"] == "Linear Kalman Filter", "mae_2d_m"].iloc[0])
        df["mae_change_vs_linear_pct"] = ((df["mae_2d_m"] - baseline_mae) / max(baseline_mae, 1e-9)) * 100.0
        return df

    def create_plots(self, traj_df: pd.DataFrame, metrics: dict, stem: str) -> dict[str, str]:
        output_paths: dict[str, str] = {}

        has_gt = traj_df[["gt_pos_x_m", "gt_pos_y_m"]].notna().all(axis=1).any()
        time_s = traj_df["time_s"].to_numpy(dtype=float)

        # Figure 1: position over time
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        if has_gt:
            axes[0].plot(time_s, traj_df["gt_pos_x_m"], label="Ground Truth", linewidth=2)
        axes[0].plot(time_s, traj_df["kf_pos_x_m"], linestyle="--", label="Linear KF")
        axes[0].plot(time_s, traj_df["bayes_pos_x_m"], linestyle="--", label="Bayesian")
        axes[0].plot(time_s, traj_df["particle_pos_x_m"], linestyle="--", label="Particle")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("X position (m)")
        axes[0].set_title("X Position Over Time")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        if has_gt:
            axes[1].plot(time_s, traj_df["gt_pos_y_m"], label="Ground Truth", linewidth=2)
        axes[1].plot(time_s, traj_df["kf_pos_y_m"], linestyle="--", label="Linear KF")
        axes[1].plot(time_s, traj_df["bayes_pos_y_m"], linestyle="--", label="Bayesian")
        axes[1].plot(time_s, traj_df["particle_pos_y_m"], linestyle="--", label="Particle")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Y position (m)")
        axes[1].set_title("Y Position Over Time")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        plt.tight_layout()
        pos_path = self.figures_dir / f"{stem}_position_over_time.png"
        plt.savefig(pos_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        output_paths["position_over_time"] = str(pos_path)

        # Figure 2: 2D trajectories
        fig, ax = plt.subplots(figsize=(10, 6))
        if has_gt:
            ax.plot(traj_df["gt_pos_x_m"], traj_df["gt_pos_y_m"], linewidth=2.5, label="Ground Truth")
            ax.plot(traj_df["gt_pos_x_m"].iloc[0], traj_df["gt_pos_y_m"].iloc[0], marker="o", markersize=10, linestyle="None", label="Start")
            ax.plot(traj_df["gt_pos_x_m"].iloc[-1], traj_df["gt_pos_y_m"].iloc[-1], marker="*", markersize=14, linestyle="None", label="End")
        ax.plot(traj_df["kf_pos_x_m"], traj_df["kf_pos_y_m"], linestyle="--", label="Linear KF")
        ax.plot(traj_df["bayes_pos_x_m"], traj_df["bayes_pos_y_m"], linestyle="--", label="Bayesian")
        ax.plot(traj_df["particle_pos_x_m"], traj_df["particle_pos_y_m"], linestyle="--", label="Particle")
        ax.set_xlabel("X position (m)")
        ax.set_ylabel("Y position (m)")
        ax.set_title("2D Trajectory Comparison")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.legend()
        plt.tight_layout()
        traj_path = self.figures_dir / f"{stem}_trajectory_comparison.png"
        plt.savefig(traj_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        output_paths["trajectory_comparison"] = str(traj_path)

        # Figure 3: error comparison
        if has_gt:
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
            error_specs = [
                ("kf_error_2d_m", "Linear KF Error", metrics["linear_kalman_filter"]["mae_2d_m"]),
                ("bayes_error_2d_m", "Bayesian Error", metrics["bayesian_filter"]["mae_2d_m"]),
                ("particle_error_2d_m", "Particle Error", metrics["particle_filter"]["mae_2d_m"]),
            ]
            for ax, (col, title, mae) in zip(axes, error_specs):
                ax.plot(time_s, traj_df[col], linewidth=1.5)
                ax.axhline(mae, linestyle="--", linewidth=1.5, label=f"MAE={mae:.4f} m")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("2D error (m)")
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.legend()
            plt.tight_layout()
            err_path = self.figures_dir / f"{stem}_error_comparison.png"
            plt.savefig(err_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            output_paths["error_comparison"] = str(err_path)

        # Figure 4: compact summary bar chart
        summary_df = self._build_summary_table(metrics)
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(summary_df))
        width = 0.35
        ax.bar(x - width / 2, summary_df["mae_2d_m"], width=width, label="MAE")
        ax.bar(x + width / 2, summary_df["rmse_2d_m"], width=width, label="RMSE")
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df["method"], rotation=10)
        ax.set_ylabel("Error (m)")
        ax.set_title("Method Comparison Summary")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        summary_fig_path = self.figures_dir / f"{stem}_summary_bars.png"
        plt.savefig(summary_fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        output_paths["summary_bars"] = str(summary_fig_path)

        return output_paths

    def evaluate(self, stem: Optional[str] = None) -> dict:
        traj_path, metrics_path = self._resolve_input_files(stem)
        traj_df = pd.read_csv(traj_path)
        metrics = self._load_metrics(metrics_path)
        summary_df = self._build_summary_table(metrics)

        stem_name = traj_path.stem.replace("_trajectories", "")
        figure_paths = self.create_plots(traj_df, metrics, stem_name)

        summary_csv_path = self.results_dir / f"{stem_name}_summary_table.csv"
        summary_df.to_csv(summary_csv_path, index=False)

        return {
            "trajectory_file": str(traj_path),
            "metrics_file": str(metrics_path),
            "summary_table_file": str(summary_csv_path),
            "figure_paths": figure_paths,
            "summary_table": summary_df,
            "metrics": metrics,
        }

    def print_summary(self, result: dict) -> None:
        print("=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print("Trajectory file:", result["trajectory_file"])
        print("Metrics file:", result["metrics_file"])
        print("Summary table file:", result["summary_table_file"])
        print()
        print(result["summary_table"].to_string(index=False))
        print()
        print("Generated figures:")
        for key, path in result["figure_paths"].items():
            print(f"- {key}: {path}")


if __name__ == "__main__":
    evaluator = ResultsEvaluator()
    result = evaluator.evaluate()
    evaluator.print_summary(result)
