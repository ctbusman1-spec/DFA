from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from filters.discrete_bayes_filter import DiscreteBayesFilter
from filters.particle_filter import ParticleFilter
from mapping.floor_map import FloorMap
from sensors.imu_reader import LiveIMUReader, OfflineIMUReader
from sensors.stride_detector import LiveStrideDetector
from utils.data_logger import DataLogger
from utils.plotting import save_comparison_plot, save_run_plot


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path):
    return yaml.safe_load(Path(path).read_text())


def ensure_map(cfg):
    map_path = PROJECT_ROOT / cfg["map"]["file"]
    if map_path.exists():
        return FloorMap.load(map_path)
    fmap = FloorMap.create_football_field(
        width_m=cfg["map"]["width_m"],
        height_m=cfg["map"]["height_m"],
        scale_m_per_cell=cfg["map"]["scale_m_per_cell"],
        sigma_cells=cfg["map"]["sigma_cells"],
        edge_decay=cfg["map"]["edge_decay"],
        center_boost=cfg["map"]["center_boost"],
    )
    fmap.save(map_path)
    return fmap


def build_filter(filter_name, floor_map, cfg):
    if filter_name == "particle":
        return ParticleFilter(floor_map, cfg["particle_filter"], cfg["motion_model"], cfg["initial_state"], rng=cfg.get("random_seed", 42))
    if filter_name == "discrete_bayes":
        return DiscreteBayesFilter(floor_map, cfg["bayes_filter"], cfg["motion_model"], cfg["initial_state"])
    raise ValueError(f"Unknown filter type: {filter_name}")


def run_offline(filter_name: str, cfg: dict, floor_map: FloorMap):
    od = cfg["offline_data"]
    reader = OfflineIMUReader(
        csv_file=od["csv_file"],
        heading_col=od["heading_column"],
        step_col=od["step_column"],
        dt_col=od["dt_column"],
        timestamp_col=od["timestamp_column"],
        fixed_step_length_m=od["fixed_step_length_m"],
    )
    model = build_filter(filter_name, floor_map, cfg)
    logger = DataLogger(PROJECT_ROOT / cfg["experiment"]["output_dir"])
    occupancy = np.zeros(floor_map.shape, dtype=float)
    started = time.perf_counter()

    for i, event in enumerate(reader.step_events()):
        result = model.update_step(event["heading_change"], event["step_length_m"], event["dt"])
        occ = model.occupancy_map(floor_map.shape)
        occupancy += occ
        x, y = result["position"]
        logger.log({
            "step_idx": i,
            "timestamp": event["timestamp"],
            "dt": event["dt"],
            "heading_change_rad": event["heading_change"],
            "step_length_m": event["step_length_m"],
            "x_m": x,
            "y_m": y,
            "var_x": result["variance"][0],
            "var_y": result["variance"][1],
            "neff": result["neff"],
            "inside_map": float(floor_map.probability(x, y) > 0.0),
        })

    runtime_s = time.perf_counter() - started
    run_df = pd.DataFrame(logger.rows)
    return run_df, occupancy, runtime_s


def run_live(filter_name: str, cfg: dict, floor_map: FloorMap):
    imu = LiveIMUReader(sample_rate_hz=cfg["live_data"]["sample_rate_hz"])
    stride = LiveStrideDetector(
        imu,
        threshold_g=cfg["live_data"]["stride_threshold_g"],
        min_stride_interval_s=cfg["live_data"]["min_stride_interval_s"],
        fixed_step_length_m=cfg["live_data"]["fixed_step_length_m"],
    )
    model = build_filter(filter_name, floor_map, cfg)
    logger = DataLogger(PROJECT_ROOT / cfg["experiment"]["output_dir"])
    occupancy = np.zeros(floor_map.shape, dtype=float)
    started = time.perf_counter()

    while True:
        event = stride.wait_for_step()
        result = model.update_step(event["heading_change"], event["step_length_m"], event["dt"])
        occupancy += model.occupancy_map(floor_map.shape)
        x, y = result["position"]
        logger.log({
            "timestamp": event["timestamp"],
            "dt": event["dt"],
            "heading_change_rad": event["heading_change"],
            "step_length_m": event["step_length_m"],
            "x_m": x,
            "y_m": y,
            "var_x": result["variance"][0],
            "var_y": result["variance"][1],
            "neff": result["neff"],
            "inside_map": float(floor_map.probability(x, y) > 0.0),
        })
        print(f"{filter_name}: x={x:.2f} y={y:.2f} neff={result['neff']:.1f}")


def summarize_run(model_name: str, run_df: pd.DataFrame, runtime_s: float):
    if run_df.empty:
        return {
            "model": model_name,
            "n_steps": 0,
            "inside_map_ratio": 0.0,
            "path_length_m": 0.0,
            "runtime_s": runtime_s,
        }
    dx = run_df["x_m"].diff().fillna(0.0)
    dy = run_df["y_m"].diff().fillna(0.0)
    return {
        "model": model_name,
        "n_steps": int(len(run_df)),
        "inside_map_ratio": float(run_df["inside_map"].mean()),
        "path_length_m": float(np.sqrt(dx**2 + dy**2).sum()),
        "runtime_s": float(runtime_s),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "system_config.yaml"))
    parser.add_argument("--filter", choices=["particle", "discrete_bayes"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.filter is not None:
        cfg["filter_type"] = args.filter

    floor_map = ensure_map(cfg)

    if cfg["mode"] == "offline":
        models = [cfg["filter_type"]]
        if cfg["experiment"].get("compare_models", False):
            models = ["particle", "discrete_bayes"]

        summaries = []
        for model_name in models:
            run_df, occupancy, runtime_s = run_offline(model_name, cfg, floor_map)
            out_dir = PROJECT_ROOT / cfg["experiment"]["output_dir"]
            csv_path = out_dir / f"{cfg['experiment']['name']}_{model_name}.csv"
            png_path = out_dir / f"{cfg['experiment']['name']}_{model_name}.png"
            out_dir.mkdir(parents=True, exist_ok=True)
            run_df.to_csv(csv_path, index=False)
            save_run_plot(run_df, floor_map, occupancy, png_path, title=f"{model_name} trajectory + heat map")
            summaries.append(summarize_run(model_name, run_df, runtime_s))
            print(f"Saved {csv_path.name} and {png_path.name}")

        summary_df = pd.DataFrame(summaries)
        summary_path = PROJECT_ROOT / cfg["experiment"]["output_dir"] / f"{cfg['experiment']['name']}_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        save_comparison_plot(summary_df, PROJECT_ROOT / cfg["experiment"]["output_dir"] / f"{cfg['experiment']['name']}_comparison.png")
        print("\nSummary")
        print(summary_df.to_string(index=False))
    else:
        run_live(cfg["filter_type"], cfg, floor_map)


if __name__ == "__main__":
    main()
