from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def infer_group(summary_path: Path) -> str:
    return summary_path.parent.name


def infer_run_name(summary_path: Path) -> str:
    return summary_path.stem.replace("_summary", "")


def load_ground_truth(gt_path: Path | None) -> pd.DataFrame | None:
    if gt_path is None or not gt_path.exists():
        return None

    if gt_path.suffix.lower() != ".csv":
        raise ValueError("Ground truth file must be a CSV file.")

    df = pd.read_csv(gt_path)
    required = {"group", "run_name"}
    if not required.issubset(df.columns):
        raise ValueError(f"Ground truth CSV must contain columns: {sorted(required)}")

    return df


def merge_ground_truth(summary_df: pd.DataFrame, gt_df: pd.DataFrame | None) -> pd.DataFrame:
    if gt_df is None:
        return summary_df

    out = summary_df.merge(gt_df, on=["group", "run_name"], how="left")

    if "expected_path_length_m" in out.columns:
        out["path_length_abs_error_m"] = (
            out["path_length_m"] - out["expected_path_length_m"]
        ).abs()

    if {"expected_final_x_m", "expected_final_y_m"}.issubset(out.columns):
        out["endpoint_error_m"] = (
            (out["final_x_m"] - out["expected_final_x_m"]) ** 2
            + (out["final_y_m"] - out["expected_final_y_m"]) ** 2
        ) ** 0.5

    return out


def build_summary(experiments_dir: Path, gt_path: Path | None = None):
    summary_files = sorted(experiments_dir.rglob("*_summary.csv"))
    if not summary_files:
        raise FileNotFoundError(f"No *_summary.csv files found under {experiments_dir}")

    frames = []
    for sf in summary_files:
        df = pd.read_csv(sf)
        df["group"] = infer_group(sf)
        df["run_name"] = infer_run_name(sf)
        df["summary_file"] = str(sf)
        frames.append(df)

    summary_df = pd.concat(frames, ignore_index=True)

    gt_df = load_ground_truth(gt_path)
    summary_df = merge_ground_truth(summary_df, gt_df)

    agg_map = {
        "n_steps": "mean",
        "walkable_ratio": "mean",
        "path_length_m": "mean",
        "runtime_s": "mean",
        "runtime_ms_per_step": "mean",
        "mean_neff": "mean",
        "final_x_m": "mean",
        "final_y_m": "mean",
    }
    for optional in ["path_length_abs_error_m", "endpoint_error_m"]:
        if optional in summary_df.columns:
            agg_map[optional] = "mean"

    grouped_df = (
        summary_df.groupby(["group", "model"], as_index=False)
        .agg(agg_map)
        .sort_values(["group", "model"])
    )

    # Best model per run:
    # first prefer higher walkable ratio, then lower path-length error if available,
    # then lower runtime per step
    best_df = summary_df.copy()
    sort_cols = ["walkable_ratio"]
    ascending = [False]

    if "path_length_abs_error_m" in best_df.columns:
        sort_cols.append("path_length_abs_error_m")
        ascending.append(True)

    sort_cols.append("runtime_ms_per_step")
    ascending.append(True)

    best_df = best_df.sort_values(sort_cols, ascending=ascending)
    best_df = best_df.groupby(["group", "run_name"], as_index=False).first()

    return summary_df, grouped_df, best_df


def build_markdown(summary_df: pd.DataFrame, grouped_df: pd.DataFrame, best_df: pd.DataFrame) -> str:
    lines = []
    lines.append("# Final Results Summary")
    lines.append("")
    lines.append("## Overall interpretation")
    lines.append("")

    if not grouped_df.empty:
        for group in grouped_df["group"].dropna().unique():
            sub = grouped_df[grouped_df["group"] == group].copy()
            lines.append(f"### {group}")

            bayes = sub[sub["model"] == "discrete_bayes"]
            part = sub[sub["model"] == "particle"]

            if not bayes.empty and not part.empty:
                bwr = bayes.iloc[0]["walkable_ratio"]
                pwr = part.iloc[0]["walkable_ratio"]
                brt = bayes.iloc[0]["runtime_ms_per_step"]
                prt = part.iloc[0]["runtime_ms_per_step"]
                winner = "discrete_bayes" if bwr >= pwr else "particle"

                lines.append(f"- Higher mean walkable ratio: **{winner}** ({bwr:.3f} vs {pwr:.3f}).")
                lines.append(f"- Runtime per step: discrete_bayes **{brt:.2f} ms**, particle **{prt:.2f} ms**.")
            lines.append("")

    lines.append("## Best model per run")
    lines.append("")

    cols = ["group", "run_name", "model", "walkable_ratio", "path_length_m", "runtime_ms_per_step"]
    extra = [c for c in ["path_length_abs_error_m", "endpoint_error_m"] if c in best_df.columns]
    cols += extra

    lines.append("```text")
    lines.append(best_df[cols].to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Still runs should be used mainly for noise / false-step / bias checks, not as trajectory-performance cases.")
    lines.append("- Walk runs are the main baseline for step-length and stability.")
    lines.append("- Short-turn / walk-turn runs are the most informative for heading behavior and corner handling.")
    lines.append("- 180-turn runs are strongest as limitation or challenge cases, not as primary performance figures.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-dir", default="../data/experiments")
    parser.add_argument("--output-dir", default="../data/experiments/outputs")
    parser.add_argument("--ground-truth-csv", default=None)
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_path = Path(args.ground_truth_csv).resolve() if args.ground_truth_csv else None

    summary_df, grouped_df, best_df = build_summary(experiments_dir, gt_path)

    summary_csv = output_dir / "final_results_summary.csv"
    grouped_csv = output_dir / "final_results_grouped_summary.csv"
    best_csv = output_dir / "final_results_best_per_run.csv"
    md_path = output_dir / "final_results_analysis.md"

    summary_df.to_csv(summary_csv, index=False)
    grouped_df.to_csv(grouped_csv, index=False)
    best_df.to_csv(best_csv, index=False)
    md_path.write_text(build_markdown(summary_df, grouped_df, best_df), encoding="utf-8")

    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {grouped_csv}")
    print(f"Wrote: {best_csv}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()