from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from imu.filters.discrete_bayes_filter import DiscreteBayesFilter
from imu.filters.particle_filter import ParticleFilter
from imu.mapping.floor_map import FloorMap
from imu.sensors.imu_reader import LiveIMUReader, OfflineIMUReader
from imu.sensors.stride_detector import LiveStrideDetector
from utils.config_utils import load_yaml
from utils.data_logger import DataLogger
from utils.paths import resolve_project_path
from utils.plotting import save_comparison_plot, save_run_plot


def ensure_map(cfg):
    map_file = resolve_project_path(cfg["map"]["file"])
    floor_map = FloorMap.from_pickle(map_file)
    print(f"Loaded floor map from: {map_file}")
    print(f"Map size [m]: {floor_map.width_m:.2f} x {floor_map.height_m:.2f}")
    print(f"Resolution [m/cell]: {floor_map.scale_m_per_cell:.3f}")
    print(f"Origin [m]: ({floor_map.origin_x_m:.2f}, {floor_map.origin_y_m:.2f})")
    return floor_map


def build_filter(filter_name, floor_map, cfg, initial_state_override=None):
    initial_state = dict(cfg["initial_state"])
    if initial_state_override is not None:
        initial_state.update(initial_state_override)

    if filter_name == "particle":
        return ParticleFilter(
            floor_map,
            cfg["particle_filter"],
            cfg["motion_model"],
            initial_state,
            rng=cfg.get("random_seed", 42),
        )
    if filter_name == "discrete_bayes":
        return DiscreteBayesFilter(
            floor_map,
            cfg["bayes_filter"],
            cfg["motion_model"],
            initial_state,
        )
    raise ValueError(f"Unknown filter type: {filter_name}")


def get_map_shape(floor_map: FloorMap):
    return floor_map.probability_map.shape


def safe_walkable(floor_map: FloorMap, x: float, y: float) -> float:
    try:
        return float(floor_map.is_walkable(x, y))
    except Exception:
        return 0.0


def safe_probability(floor_map: FloorMap, x: float, y: float) -> float:
    try:
        return float(floor_map.get_probability(x, y))
    except Exception:
        return 0.0


def run_offline(filter_name: str, cfg: dict, floor_map: FloorMap):
    od = cfg["offline_data"]
    init_cfg = cfg.get("initialization", {})
    csv_file = resolve_project_path(od["csv_file"])

    reader = OfflineIMUReader(
        csv_file=csv_file,
        heading_col=od["heading_column"],
        step_col=od["step_column"],
        dt_col=od["dt_column"],
        timestamp_col=od["timestamp_column"],
        fixed_step_length_m=od["fixed_step_length_m"],
        gyro_col=od.get("gyro_column", "gyro_z_rads"),
        gyro_bias_estimate_seconds=od.get("gyro_bias_estimate_seconds", 1.5),
    )

    initial_state_override = None
    if init_cfg.get("use_compass_for_initial_heading", False):
        measured_heading = reader.estimate_initial_heading(
            average_seconds=float(init_cfg.get("compass_average_seconds", 1.5))
        )
        map_heading_offset = float(init_cfg.get("map_heading_offset_rad", 0.0))
        initial_heading = measured_heading + map_heading_offset

        initial_state_override = {"heading": initial_heading}

        print(
            f"{filter_name}: initial heading from compass/log = "
            f"{measured_heading:.3f} rad, map offset = {map_heading_offset:.3f}, "
            f"used = {initial_heading:.3f} rad"
        )

    model = build_filter(filter_name, floor_map, cfg, initial_state_override=initial_state_override)

    logger = DataLogger(resolve_project_path(cfg["experiment"]["output_dir"]))
    occupancy = np.zeros(get_map_shape(floor_map), dtype=float)
    started = time.perf_counter()

    for i, event in enumerate(reader.step_events()):
        heading_change = float(event["heading_change"])
        step_length = float(event["step_length_m"])
        dt = float(event["dt"])

        if not np.isfinite(heading_change):
            heading_change = 0.0
        if not np.isfinite(step_length):
            step_length = 0.0
        if not np.isfinite(dt) or dt <= 0:
            dt = 1e-3

        result = model.update_step(heading_change, step_length, dt)
        occupancy += model.occupancy_map(get_map_shape(floor_map))
        x, y = result["position"]

        logger.log({
            "step_idx": i,
            "timestamp": event["timestamp"],
            "dt": dt,
            "heading_change_rad": heading_change,
            "heading_rad": result["heading_rad"],
            "step_length_m": step_length,
            "x_m": x,
            "y_m": y,
            "var_x": result["variance"][0],
            "var_y": result["variance"][1],
            "neff": result["neff"],
            "map_probability": safe_probability(floor_map, x, y),
            "walkable": safe_walkable(floor_map, x, y),
        })

    runtime_s = time.perf_counter() - started
    run_df = logger.to_frame()
    return run_df, occupancy, runtime_s


def run_live(filter_name: str, cfg: dict, floor_map: FloorMap):
    ld = cfg["live_data"]
    init_cfg = cfg.get("initialization", {})

    imu = LiveIMUReader(sample_rate_hz=ld["sample_rate_hz"])
    stride = LiveStrideDetector(
        imu,
        threshold_g=ld["stride_threshold_g"],
        min_stride_interval_s=ld["min_stride_interval_s"],
        fixed_step_length_m=ld["fixed_step_length_m"],
    )

    initial_state_override = None
    if init_cfg.get("use_compass_for_initial_heading", False):
        measured_heading = imu.estimate_initial_heading(
            average_seconds=float(init_cfg.get("compass_average_seconds", 1.5))
        )
        map_heading_offset = float(init_cfg.get("map_heading_offset_rad", 0.0))
        initial_heading = measured_heading + map_heading_offset

        initial_state_override = {"heading": initial_heading}

        print(
            f"{filter_name}: live initial heading from compass = {measured_heading:.3f} rad, "
            f"map offset = {map_heading_offset:.3f}, used = {initial_heading:.3f} rad"
        )

    model = build_filter(filter_name, floor_map, cfg, initial_state_override=initial_state_override)
    logger = DataLogger(resolve_project_path(cfg["experiment"]["output_dir"]))
    occupancy = np.zeros(get_map_shape(floor_map), dtype=float)
    started = time.perf_counter()
    step_idx = 0

    try:
        while True:
            event = stride.wait_for_step()

            heading_change = float(event["heading_change"])
            step_length = float(event["step_length_m"])
            dt = float(event["dt"])

            if not np.isfinite(heading_change):
                heading_change = 0.0
            if not np.isfinite(step_length):
                step_length = 0.0
            if not np.isfinite(dt) or dt <= 0:
                dt = 1e-3

            result = model.update_step(heading_change, step_length, dt)
            occupancy += model.occupancy_map(get_map_shape(floor_map))
            x, y = result["position"]

            logger.log({
                "step_idx": step_idx,
                "timestamp": event["timestamp"],
                "dt": dt,
                "heading_change_rad": heading_change,
                "heading_rad": result["heading_rad"],
                "step_length_m": step_length,
                "x_m": x,
                "y_m": y,
                "var_x": result["variance"][0],
                "var_y": result["variance"][1],
                "neff": result["neff"],
                "map_probability": safe_probability(floor_map, x, y),
                "walkable": safe_walkable(floor_map, x, y),
            })

            print(
                f"{filter_name}: step={step_idx:03d} "
                f"x={x:.2f} y={y:.2f} "
                f"heading={result['heading_rad']:.2f} "
                f"neff={result['neff']:.1f}"
            )
            step_idx += 1

    except KeyboardInterrupt:
        print("\nLive run stopped by user.")

    runtime_s = time.perf_counter() - started
    run_df = logger.to_frame()
    return run_df, occupancy, runtime_s


def summarize_run(model_name: str, run_df: pd.DataFrame, runtime_s: float):
    if run_df.empty:
        return {
            "model": model_name,
            "n_steps": 0,
            "walkable_ratio": 0.0,
            "path_length_m": 0.0,
            "runtime_s": runtime_s,
            "runtime_ms_per_step": 0.0,
            "mean_neff": 0.0,
            "final_x_m": np.nan,
            "final_y_m": np.nan,
        }

    dx = run_df["x_m"].diff().fillna(0.0)
    dy = run_df["y_m"].diff().fillna(0.0)
    n_steps = int(len(run_df))

    return {
        "model": model_name,
        "n_steps": n_steps,
        "walkable_ratio": float(run_df["walkable"].mean()),
        "path_length_m": float(np.sqrt(dx ** 2 + dy ** 2).sum()),
        "runtime_s": float(runtime_s),
        "runtime_ms_per_step": float(1000.0 * runtime_s / max(n_steps, 1)),
        "mean_neff": float(run_df["neff"].mean()) if "neff" in run_df.columns else np.nan,
        "final_x_m": float(run_df["x_m"].iloc[-1]),
        "final_y_m": float(run_df["y_m"].iloc[-1]),
    }


def save_outputs(model_name: str, cfg: dict, floor_map: FloorMap, run_df: pd.DataFrame, occupancy, runtime_s: float):
    out_dir = resolve_project_path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{cfg['experiment']['name']}_{model_name}.csv"
    png_path = out_dir / f"{cfg['experiment']['name']}_{model_name}.png"

    run_df.to_csv(csv_path, index=False)
    save_run_plot(
        run_df,
        floor_map,
        occupancy,
        png_path,
        title=f"{model_name} trajectory over floorplan PDF",
    )
    print(f"Saved {csv_path.name} and {png_path.name}")

    return summarize_run(model_name, run_df, runtime_s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/system_config.yaml")
    parser.add_argument("--filter", choices=["particle", "discrete_bayes"], default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.filter is not None:
        cfg["filter_type"] = args.filter

    floor_map = ensure_map(cfg)

    models = [cfg["filter_type"]]
    if cfg["experiment"].get("compare_models", False):
        models = ["particle", "discrete_bayes"]

    summaries = []

    if cfg["mode"] == "offline":
        for model_name in models:
            run_df, occupancy, runtime_s = run_offline(model_name, cfg, floor_map)
            summaries.append(save_outputs(model_name, cfg, floor_map, run_df, occupancy, runtime_s))

    else:
        for model_name in models:
            run_df, occupancy, runtime_s = run_live(model_name, cfg, floor_map)
            summaries.append(save_outputs(model_name, cfg, floor_map, run_df, occupancy, runtime_s))

    out_dir = resolve_project_path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summaries)
    summary_path = out_dir / f"{cfg['experiment']['name']}_summary.csv"
    summary_png = out_dir / f"{cfg['experiment']['name']}_comparison.png"

    summary_df.to_csv(summary_path, index=False)
    if len(summary_df) > 0:
        save_comparison_plot(summary_df, summary_png)

    print("\nSummary")
    if summary_df.empty:
        print("No results.")
    else:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()