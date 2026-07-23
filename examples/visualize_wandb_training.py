#!/usr/bin/env python3

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wandb.proto import wandb_internal_pb2 as wandb_internal
from wandb.sdk.internal.datastore import DataStore


PLOT_GROUPS = {
    "01_buffer_success": [
        "buffer/actor_saved_trajectories",
        "buffer/actor_saved_success_trajectories",
        "buffer/actor_saved_failure_trajectories",
        "buffer/actor_saved_intervention_trajectories",
        "buffer/actor_saved_policy_trajectories",
        "buffer/actor_saved_transitions",
        "success_rate",
        "intervention_rate",
    ],
    "02_last_trajectory": [
        "buffer/last_trajectory_success",
        "last_success_roll10",
        "buffer/last_trajectory_intervention",
        "buffer/last_trajectory_length",
        "human_intervention/total_time_s",
        "human_intervention/last_trajectory_time_s",
    ],
    "03_training_core": [
        "critic/rewards",
        "critic/critic_loss",
        "critic/predicted_qs",
        "critic/target_qs",
        "actor/actor_loss",
        "actor/entropy",
        "actor/temperature",
        "temperature/temperature_loss",
    ],
    "04_causal_mask": [
        "causal_mask/causal_model_ready",
        "causal_mask/causal_model_loss",
        "causal_mask/causal_val_nll",
        "causal_mask/causal_val_mse",
        "causal_mask/masked_fraction_last_batch",
        "causal_mask/selected_latent_ratio",
        "causal_mask/selected_cmi_mean",
        "causal_mask/latent_delta_l1",
    ],
    "05_causal_entropy": [
        "causal_entropy/active",
        "causal_entropy/disabled_by_policy_success",
        "causal_entropy/policy_success_streak",
        "causal_entropy/weight_std",
        "causal_entropy/action_weight_dim_0",
        "causal_entropy/action_weight_dim_1",
        "causal_entropy/action_weight_dim_2",
        "actor/causal_entropy_delta",
    ],
    "06_timers": [
        "timer/train",
        "timer/sample_train_batch",
        "timer/train_critics",
        "timer/causal_model_update",
        "timer/causal_weight_computation",
    ],
}


def find_wandb_run_dir(wandb_dir: Path) -> Path:
    if not wandb_dir.exists():
        raise FileNotFoundError(f"W&B directory does not exist: {wandb_dir}")

    if wandb_dir.is_file() and wandb_dir.suffix == ".wandb":
        return wandb_dir.parent

    if any(wandb_dir.glob("*.wandb")):
        return wandb_dir

    latest = wandb_dir / "latest-run"
    if latest.exists():
        return latest.resolve()

    run_dirs = sorted(
        [p for p in wandb_dir.glob("run-*") if p.is_dir() and any(p.glob("*.wandb"))],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if run_dirs:
        return run_dirs[0]

    raise FileNotFoundError(f"No local W&B run found under: {wandb_dir}")


def find_wandb_file(run_dir: Path) -> Path:
    wandb_files = sorted(run_dir.glob("*.wandb"))
    if not wandb_files:
        raise FileNotFoundError(f"No .wandb file found in: {run_dir}")
    if len(wandb_files) > 1:
        wandb_files = sorted(wandb_files, key=lambda p: p.stat().st_mtime, reverse=True)
    return wandb_files[0]


def load_history(wandb_file: Path) -> pd.DataFrame:
    datastore = DataStore()
    datastore.open_for_scan(str(wandb_file))
    rows = []
    try:
        while True:
            try:
                data = datastore.scan_data()
            except AssertionError as exc:
                print(f"Stopped at a corrupt W&B record: {exc}")
                break
            if data is None:
                break
            record = wandb_internal.Record()
            record.ParseFromString(data)
            if record.WhichOneof("record_type") != "history":
                continue

            row = {}
            for item in record.history.item:
                try:
                    row[item.key] = json.loads(item.value_json)
                except json.JSONDecodeError:
                    row[item.key] = item.value_json
            if row:
                rows.append(row)
    finally:
        datastore.close()

    if not rows:
        raise ValueError(f"No history records found in: {wandb_file}")

    df = pd.DataFrame(rows)
    df.insert(0, "_row", np.arange(len(df)))
    for col in df.columns:
        if col == "_row":
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any() or df[col].notna().sum() == 0:
            df[col] = converted
    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {
        "buffer/actor_saved_success_trajectories",
        "buffer/actor_saved_trajectories",
    }.issubset(df.columns):
        total = pd.to_numeric(df["buffer/actor_saved_trajectories"], errors="coerce")
        success = pd.to_numeric(
            df["buffer/actor_saved_success_trajectories"], errors="coerce"
        )
        df["success_rate"] = success / total.replace(0, np.nan)

    if {
        "buffer/actor_saved_intervention_trajectories",
        "buffer/actor_saved_trajectories",
    }.issubset(df.columns):
        total = pd.to_numeric(df["buffer/actor_saved_trajectories"], errors="coerce")
        intvn = pd.to_numeric(
            df["buffer/actor_saved_intervention_trajectories"], errors="coerce"
        )
        df["intervention_rate"] = intvn / total.replace(0, np.nan)

    if "buffer/last_trajectory_success" in df.columns:
        success = pd.to_numeric(df["buffer/last_trajectory_success"], errors="coerce")
        df["last_success_roll10"] = success.rolling(10, min_periods=1).mean()

    return df


def numeric_series(df: pd.DataFrame, key: str) -> pd.Series:
    return pd.to_numeric(df[key], errors="coerce")


def choose_x_column(df: pd.DataFrame) -> str:
    if "_step" in df.columns and numeric_series(df, "_step").notna().any():
        return "_step"
    return "_row"


def plot_group(
    df: pd.DataFrame,
    x_col: str,
    metrics: Iterable[str],
    title: str,
    output_path: Path,
) -> bool:
    available = [m for m in metrics if m in df.columns and numeric_series(df, m).notna().any()]
    if not available:
        return False

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(12, max(2.4 * n, 4)), sharex=True)
    if n == 1:
        axes = [axes]

    x = numeric_series(df, x_col) if x_col in df.columns else df["_row"]
    for ax, metric in zip(axes, available):
        y = numeric_series(df, metric)
        mask = x.notna() & y.notna()
        ax.plot(x[mask], y[mask], linewidth=1.8)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel(x_col)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def safe_metric_filename(metric: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", metric)
    return name.strip("_") or "metric"


def plot_single_metric(
    df: pd.DataFrame,
    x_col: str,
    metric: str,
    output_path: Path,
) -> bool:
    if metric not in df.columns:
        return False
    x = numeric_series(df, x_col) if x_col in df.columns else df["_row"]
    y = numeric_series(df, metric)
    mask = x.notna() & y.notna()
    if not mask.any():
        return False

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(x[mask], y[mask], linewidth=1.9)
    ax.set_title(metric)
    ax.set_xlabel(x_col)
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return True


def plot_individual_metrics(df: pd.DataFrame, x_col: str, output_dir: Path) -> list[str]:
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    ordered_metrics = []
    seen = set()
    for metrics in PLOT_GROUPS.values():
        for metric in metrics:
            if metric not in seen:
                ordered_metrics.append(metric)
                seen.add(metric)

    plotted = []
    for metric in ordered_metrics:
        filename = f"{safe_metric_filename(metric)}.png"
        if plot_single_metric(df, x_col, metric, metrics_dir / filename):
            plotted.append(f"metrics/{filename}")
    return plotted


def load_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "files" / "wandb-summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def last_valid_values(df: pd.DataFrame, keys: Iterable[str]) -> dict:
    values = {}
    for key in keys:
        if key not in df.columns:
            continue
        series = df[key].dropna()
        if not series.empty:
            value = series.iloc[-1]
            if isinstance(value, np.generic):
                value = value.item()
            values[key] = value
    return values


def write_report(
    output_dir: Path,
    wandb_dir: Path,
    run_dir: Path,
    wandb_file: Path,
    df: pd.DataFrame,
    plotted: list[str],
    individual_plotted: list[str],
) -> None:
    key_metrics = [
        "buffer/actor_saved_trajectories",
        "buffer/actor_saved_success_trajectories",
        "buffer/actor_saved_failure_trajectories",
        "buffer/actor_saved_intervention_trajectories",
        "buffer/actor_saved_transitions",
        "success_rate",
        "intervention_rate",
        "critic/rewards",
        "actor/actor_loss",
        "critic/critic_loss",
        "causal_mask/causal_model_loss",
        "causal_mask/causal_val_nll",
        "causal_mask/causal_val_mse",
        "causal_mask/masked_fraction_last_batch",
        "causal_entropy/policy_success_streak",
        "causal_entropy/disabled_by_policy_success",
    ]
    values = last_valid_values(df, key_metrics)

    lines = [
        f"wandb_dir: {wandb_dir}",
        f"run_dir: {run_dir}",
        f"wandb_file: {wandb_file}",
        f"history_rows: {len(df)}",
        f"history_columns: {len(df.columns)}",
        "",
        "final_values:",
    ]
    for key, value in values.items():
        lines.append(f"  {key}: {value}")
    lines += ["", "plots:"]
    for name in plotted:
        lines.append(f"  {name}.png")
    lines += ["", "individual_metric_plots:"]
    for name in individual_plotted:
        lines.append(f"  {name}")
    lines.append("")

    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def default_output_dir(wandb_arg: Path, run_dir: Path) -> Path:
    resolved_arg = wandb_arg.resolve()
    if resolved_arg.is_dir() and resolved_arg.name == "wandb":
        return resolved_arg.parent / "training_visualization"
    if (
        resolved_arg.is_dir()
        and resolved_arg.name.startswith("run-")
        and resolved_arg.parent.name == "wandb"
    ):
        return resolved_arg.parent.parent / "training_visualization"
    if (
        resolved_arg.is_file()
        and resolved_arg.suffix == ".wandb"
        and resolved_arg.parent.parent.name == "wandb"
    ):
        return resolved_arg.parent.parent.parent / "training_visualization"
    return run_dir.parent / "training_visualization"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize local W&B history from a SERL training run."
    )
    parser.add_argument(
        "--wandb_dir",
        required=True,
        type=Path,
        help="Path to a local wandb directory, run directory, or .wandb file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run_parent>/training_visualization.",
    )
    args = parser.parse_args()

    run_dir = find_wandb_run_dir(args.wandb_dir)
    wandb_file = find_wandb_file(run_dir)
    output_dir = args.output_dir or default_output_dir(args.wandb_dir, run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = add_derived_metrics(load_history(wandb_file))
    x_col = choose_x_column(df)
    df.to_csv(output_dir / "wandb_history.csv", index=False)

    plotted = []
    for name, metrics in PLOT_GROUPS.items():
        if plot_group(df, x_col, metrics, name, output_dir / f"{name}.png"):
            plotted.append(name)
    individual_plotted = plot_individual_metrics(df, x_col, output_dir)

    summary = load_summary(run_dir)
    if summary:
        (output_dir / "wandb_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
    write_report(
        output_dir,
        args.wandb_dir,
        run_dir,
        wandb_file,
        df,
        plotted,
        individual_plotted,
    )

    print(f"Read W&B run: {run_dir}")
    print(f"History rows: {len(df)}, columns: {len(df.columns)}")
    print(f"Wrote visualization to: {output_dir}")


if __name__ == "__main__":
    main()
