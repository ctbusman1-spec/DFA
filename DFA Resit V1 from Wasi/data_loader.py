from __future__ import annotations

"""
Data loading and schema harmonization for the DFA IMU assignment.

This module supports two dataset families:
1. Benchmark dataset with ground truth and full IMU-style channels.
2. Raspberry Pi logs with reduced channels and no ground truth.

All loaded datasets are converted to one common representational format so the
rest of the pipeline can work with a stable interface.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import get_config


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata describing a loaded dataset."""

    source_name: str
    dataset_type: str
    n_samples: int
    has_ground_truth: bool
    has_gyro: bool
    has_walkability: bool
    has_step_counter: bool


REQUIRED_BENCHMARK_COLUMNS = {
    "t",
    "dt",
    "acc_x_g",
    "acc_y_g",
    "acc_z_g",
    "gyro_x_rads",
    "gyro_y_rads",
    "gyro_z_rads",
    "heading_rad",
    "step",
    "step_count",
    "pos_x_m",
    "pos_y_m",
}

REQUIRED_PI_COLUMNS = {
    "timestamp",
    "dt",
    "heading_rad",
    "accel_x",
    "accel_y",
    "accel_z",
    "accel_norm",
    "step_event",
}


class DataLoader:
    """Load, validate, and harmonize supported IMU CSV datasets."""

    def __init__(self) -> None:
        self.cfg = get_config()
        self.g = self.cfg.units.gravity_mps2

    def detect_dataset_type(self, df: pd.DataFrame) -> str:
        """Infer dataset family from available columns."""
        cols = set(df.columns)

        if REQUIRED_BENCHMARK_COLUMNS.issubset(cols):
            return "benchmark"
        if REQUIRED_PI_COLUMNS.issubset(cols):
            return "pi"
        raise ValueError(
            "Unsupported dataset schema. "
            f"Found columns: {sorted(cols)}"
        )

    def load_csv(self, file_path: str | Path) -> tuple[pd.DataFrame, DatasetInfo]:
        """Load a CSV file and return a normalized dataframe plus metadata."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        raw_df = pd.read_csv(path)
        dataset_type = self.detect_dataset_type(raw_df)

        if dataset_type == "benchmark":
            norm_df = self._normalize_benchmark(raw_df)
        else:
            norm_df = self._normalize_pi(raw_df)

        self._validate_normalized(norm_df)
        info = self._build_info(path.name, dataset_type, norm_df)
        return norm_df, info

    def load_default_dataset(self) -> tuple[pd.DataFrame, DatasetInfo]:
        """Load the main benchmark dataset from config.py."""
        return self.load_csv(self.cfg.paths.data_file)

    def summarize(self, df: pd.DataFrame, info: DatasetInfo) -> Dict[str, object]:
        """Return a compact summary for logs and sanity checks."""
        return {
            "source_name": info.source_name,
            "dataset_type": info.dataset_type,
            "n_samples": info.n_samples,
            "duration_s": round(float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]), 4),
            "mean_dt_s": round(float(df["dt_s"].replace(0.0, np.nan).mean()), 6),
            "step_events": int(df["step_event"].sum()),
            "has_ground_truth": info.has_ground_truth,
            "has_gyro": info.has_gyro,
            "x_range_m": self._safe_range(df["gt_pos_x_m"]),
            "y_range_m": self._safe_range(df["gt_pos_y_m"]),
        }

    def _normalize_benchmark(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map benchmark dataset columns to the common representational format."""
        out = pd.DataFrame()

        out["time_s"] = pd.to_numeric(df["t"], errors="coerce")
        out["dt_s"] = pd.to_numeric(df["dt"], errors="coerce")

        out["acc_x_g"] = pd.to_numeric(df["acc_x_g"], errors="coerce")
        out["acc_y_g"] = pd.to_numeric(df["acc_y_g"], errors="coerce")
        out["acc_z_g"] = pd.to_numeric(df["acc_z_g"], errors="coerce")

        out["acc_x_mps2"] = out["acc_x_g"] * self.g
        out["acc_y_mps2"] = out["acc_y_g"] * self.g
        out["acc_z_mps2"] = out["acc_z_g"] * self.g
        out["acc_norm_g"] = np.sqrt((out[["acc_x_g", "acc_y_g", "acc_z_g"]] ** 2).sum(axis=1))

        out["gyro_x_rads"] = pd.to_numeric(df["gyro_x_rads"], errors="coerce")
        out["gyro_y_rads"] = pd.to_numeric(df["gyro_y_rads"], errors="coerce")
        out["gyro_z_rads"] = pd.to_numeric(df["gyro_z_rads"], errors="coerce")
        out["gyro_norm_rads"] = np.sqrt((out[["gyro_x_rads", "gyro_y_rads", "gyro_z_rads"]] ** 2).sum(axis=1))

        out["heading_rad"] = pd.to_numeric(df["heading_rad"], errors="coerce")
        out["step_event"] = pd.to_numeric(df["step"], errors="coerce").fillna(0).astype(int)
        out["step_count"] = pd.to_numeric(df["step_count"], errors="coerce").fillna(0).astype(int)

        out["gt_pos_x_m"] = pd.to_numeric(df["pos_x_m"], errors="coerce")
        out["gt_pos_y_m"] = pd.to_numeric(df["pos_y_m"], errors="coerce")
        out["gt_pos_z_m"] = self.cfg.units.floor_z_m
        out["walkable"] = pd.to_numeric(df.get("walkable", 1), errors="coerce").fillna(1).astype(int)

        out["source_dataset_type"] = "benchmark"
        out["has_ground_truth"] = True
        return self._finalize_normalized(out)

    def _normalize_pi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map Raspberry Pi logs to the common representational format."""
        out = pd.DataFrame()

        out["time_s"] = pd.to_numeric(df["timestamp"], errors="coerce")
        out["dt_s"] = pd.to_numeric(df["dt"], errors="coerce")

        out["acc_x_g"] = pd.to_numeric(df["accel_x"], errors="coerce")
        out["acc_y_g"] = pd.to_numeric(df["accel_y"], errors="coerce")
        out["acc_z_g"] = pd.to_numeric(df["accel_z"], errors="coerce")

        out["acc_x_mps2"] = out["acc_x_g"] * self.g
        out["acc_y_mps2"] = out["acc_y_g"] * self.g
        out["acc_z_mps2"] = out["acc_z_g"] * self.g
        out["acc_norm_g"] = pd.to_numeric(df["accel_norm"], errors="coerce")

        # Pi logs do not include gyroscope channels, so we keep explicit zeros.
        out["gyro_x_rads"] = 0.0
        out["gyro_y_rads"] = 0.0
        out["gyro_z_rads"] = 0.0
        out["gyro_norm_rads"] = 0.0

        out["heading_rad"] = pd.to_numeric(df["heading_rad"], errors="coerce")
        out["step_event"] = pd.to_numeric(df["step_event"], errors="coerce").fillna(0).astype(int)
        out["step_count"] = out["step_event"].cumsum().astype(int)

        # No true position labels are available in the Pi logs.
        out["gt_pos_x_m"] = np.nan
        out["gt_pos_y_m"] = np.nan
        out["gt_pos_z_m"] = np.nan
        out["walkable"] = np.nan

        out["source_dataset_type"] = "pi"
        out["has_ground_truth"] = False
        return self._finalize_normalized(out)

    def _finalize_normalized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Common cleanup after dataset-specific normalization."""
        df = df.copy()

        # Remove fully invalid rows in the main timing channel.
        df = df.dropna(subset=["time_s"]).reset_index(drop=True)

        # Fill dt where the first sample may contain a tiny or missing startup gap.
        if df["dt_s"].isna().any() or (df["dt_s"] <= 0).any():
            inferred_dt = df["time_s"].diff().fillna(df["dt_s"].median())
            inferred_dt = inferred_dt.mask(inferred_dt <= 0, df["dt_s"].median())
            df["dt_s"] = df["dt_s"].where(df["dt_s"] > 0, inferred_dt)

        df["sample_idx"] = np.arange(len(df), dtype=int)
        ordered_cols = [
            "sample_idx",
            "time_s",
            "dt_s",
            "acc_x_g",
            "acc_y_g",
            "acc_z_g",
            "acc_x_mps2",
            "acc_y_mps2",
            "acc_z_mps2",
            "acc_norm_g",
            "gyro_x_rads",
            "gyro_y_rads",
            "gyro_z_rads",
            "gyro_norm_rads",
            "heading_rad",
            "step_event",
            "step_count",
            "gt_pos_x_m",
            "gt_pos_y_m",
            "gt_pos_z_m",
            "walkable",
            "source_dataset_type",
            "has_ground_truth",
        ]
        return df[ordered_cols]

    def _validate_normalized(self, df: pd.DataFrame) -> None:
        """Basic sanity checks to fail early on broken inputs."""
        required_cols = {
            "time_s",
            "dt_s",
            "acc_x_mps2",
            "acc_y_mps2",
            "acc_z_mps2",
            "heading_rad",
            "step_event",
        }
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"Normalized dataset is missing required columns: {sorted(missing)}")

        if len(df) == 0:
            raise ValueError("Normalized dataset is empty.")

        if df["time_s"].isna().all():
            raise ValueError("All time values are NaN after normalization.")

    def _build_info(self, source_name: str, dataset_type: str, df: pd.DataFrame) -> DatasetInfo:
        """Construct metadata from the normalized dataframe."""
        has_ground_truth = df[["gt_pos_x_m", "gt_pos_y_m"]].notna().all(axis=1).any()
        has_gyro = np.abs(df[["gyro_x_rads", "gyro_y_rads", "gyro_z_rads"]]).sum().sum() > 0
        has_walkability = df["walkable"].notna().any()
        has_step_counter = df["step_count"].notna().any()
        return DatasetInfo(
            source_name=source_name,
            dataset_type=dataset_type,
            n_samples=len(df),
            has_ground_truth=bool(has_ground_truth),
            has_gyro=bool(has_gyro),
            has_walkability=bool(has_walkability),
            has_step_counter=bool(has_step_counter),
        )

    @staticmethod
    def _safe_range(series: pd.Series) -> Optional[tuple[float, float]]:
        """Return a min/max tuple when the series has valid values."""
        valid = series.dropna()
        if valid.empty:
            return None
        return round(float(valid.min()), 4), round(float(valid.max()), 4)


if __name__ == "__main__":
    loader = DataLoader()

    demo_candidates = [
        loader.cfg.paths.data_file,
        Path("/mnt/data/Real-data-set-sensor_log_4.csv"),
        Path("/mnt/data/pi_sensor_log_walk_straight.csv"),
    ]
    demo_files = [path for path in demo_candidates if Path(path).exists()]

    for file_path in demo_files:
        print("=" * 80)
        print(f"Loading: {file_path}")
        df_norm, info = loader.load_csv(file_path)
        for key, value in loader.summarize(df_norm, info).items():
            print(f"- {key}: {value}")
        print("- normalized_columns:", list(df_norm.columns))
        print(df_norm.head(2).to_string(index=False))