from __future__ import annotations

"""
Configuration for the DFA IMU assignment rebuild.

This file centralizes all experiment settings so the rest of the pipeline
can stay clean, reproducible, and easy to explain in the notebook.
Compatible with normal Python scripts, VS Code, Jupyter, and Google Colab.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


def _detect_project_root() -> Path:
    """
    Resolve a stable project root.

    - In normal .py execution, use the folder containing this file.
    - In notebooks / Colab where __file__ is not defined, use the current
      working directory.
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


PROJECT_ROOT = _detect_project_root()


@dataclass(frozen=True)
class PathsConfig:
    """Project file paths."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    data_file: Path = PROJECT_ROOT / "data" / "Real-data-set-sensor_log_4.csv"
    output_dir: Path = PROJECT_ROOT / "outputs"
    figures_dir: Path = PROJECT_ROOT / "outputs" / "figures"
    results_dir: Path = PROJECT_ROOT / "outputs" / "results"


@dataclass(frozen=True)
class DatasetSchema:
    """Mapping from CSV columns to semantic sensor fields."""

    time_col: str = "t"
    dt_col: str = "dt"

    accel_cols_g: Tuple[str, str, str] = ("acc_x_g", "acc_y_g", "acc_z_g")
    gyro_cols_rads: Tuple[str, str, str] = (
        "gyro_x_rads",
        "gyro_y_rads",
        "gyro_z_rads",
    )

    heading_col: str = "heading_rad"
    step_event_col: str = "step"
    step_count_col: str = "step_count"
    walkable_col: str = "walkable"

    gt_position_cols_m: Tuple[str, str] = ("pos_x_m", "pos_y_m")


@dataclass(frozen=True)
class UnitConfig:
    """Measurement units and conversions."""

    gravity_mps2: float = 9.81
    floor_z_m: float = 0.0


@dataclass(frozen=True)
class ZUPTConfig:
    """Parameters for simplified stationary detection."""

    window_size: int = 10
    accel_variance_threshold: float = 0.01
    gyro_variance_threshold: float = 1e-5
    velocity_threshold_mps: float = 0.05
    confirmation_samples: int = 5


@dataclass(frozen=True)
class KalmanConfig:
    """Parameters for the linear Kalman filter."""

    dt_default_s: float = 0.045
    process_noise_accel: float = 0.10
    process_noise_drift: float = 0.01
    measurement_noise_accel: float = 0.10
    measurement_noise_velocity: float = 0.05
    measurement_noise_zupt: float = 0.01


@dataclass(frozen=True)
class BayesianConfig:
    """Parameters for the non-recursive Bayesian filter."""

    process_noise: float = 0.02
    measurement_noise: float = 0.15
    window_size: int = 10
    grid_resolution_m: float = 0.10
    candidate_count: int = 1000
    mode_bandwidth_m: float = 0.20


@dataclass(frozen=True)
class ParticleConfig:
    """Parameters for the particle filter."""

    n_particles: int = 500
    process_noise: float = 0.015
    measurement_noise: float = 0.25
    resampling_threshold: float = 0.80
    dt_default_s: float = 0.045


@dataclass(frozen=True)
class FloorplanConfig:
    """Static floorplan prior parameters."""

    cell_size_cm: float = 10.0
    sigma_cells: float = 1.6
    draw_value: float = 20.0
    max_prob_cap: float = 0.5
    padding_m: float = 0.5
    mirror_vertically: bool = True
    right_corridor_drop_m: float = 0.40
    start_position_m: Tuple[float, float] = (1.0, 0.70)


@dataclass(frozen=True)
class EvaluationConfig:
    """Settings for validation and plots."""

    measurement_covariance_scale: float = 0.10
    zero_accel_update_threshold_mps2: float = 0.10
    random_seed: int = 42


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    experiment_name: str = "dfa_imu_rebuild"
    paths: PathsConfig = field(default_factory=PathsConfig)
    schema: DatasetSchema = field(default_factory=DatasetSchema)
    units: UnitConfig = field(default_factory=UnitConfig)
    zupt: ZUPTConfig = field(default_factory=ZUPTConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    bayesian: BayesianConfig = field(default_factory=BayesianConfig)
    particle: ParticleConfig = field(default_factory=ParticleConfig)
    floorplan: FloorplanConfig = field(default_factory=FloorplanConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def ensure_directories(self) -> None:
        """Create required folders if they do not exist."""
        self.paths.data_dir.mkdir(parents=True, exist_ok=True)
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths.figures_dir.mkdir(parents=True, exist_ok=True)
        self.paths.results_dir.mkdir(parents=True, exist_ok=True)

    def to_summary_dict(self) -> Dict[str, object]:
        """Compact summary useful for logs and debugging."""
        return {
            "experiment_name": self.experiment_name,
            "project_root": str(self.paths.project_root),
            "data_file": str(self.paths.data_file),
            "output_dir": str(self.paths.output_dir),
            "accel_cols_g": self.schema.accel_cols_g,
            "gyro_cols_rads": self.schema.gyro_cols_rads,
            "gt_position_cols_m": self.schema.gt_position_cols_m,
            "n_particles": self.particle.n_particles,
            "bayesian_grid_resolution_m": self.bayesian.grid_resolution_m,
            "floorplan_cell_size_cm": self.floorplan.cell_size_cm,
        }


def get_config() -> AppConfig:
    """Return the default application configuration."""
    config = AppConfig()
    config.ensure_directories()
    return config


if __name__ == "__main__":
    cfg = get_config()
    print("Loaded configuration:")
    for key, value in cfg.to_summary_dict().items():
        print(f"- {key}: {value}")
    print("\nPut your CSV here if running locally:")
    print(f"  {cfg.paths.data_file}")