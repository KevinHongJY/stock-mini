from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from d3qn_stock.algos.d3qn.trainer import Config, evaluate, load_config, save_config, train
from d3qn_stock.utils.logging import setup_run_logger
from d3qn_stock.utils.path import RunPaths


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    color: str
    overrides: tuple[tuple[str, object], ...]


DEFAULT_VARIANTS: tuple[Variant, ...] = (
    Variant(
        key="dqn",
        label="DQN",
        color="#4E79A7",
        overrides=(
            ("model.type", "mlp"),
            ("agent.double", False),
            ("agent.per_enabled", False),
            ("agent.n_step", 1),
            ("model.use_noisy", False),
        ),
    ),
    Variant(
        key="ddqn",
        label="Double DQN",
        color="#F28E2B",
        overrides=(
            ("model.type", "mlp"),
            ("agent.double", True),
            ("agent.per_enabled", False),
            ("agent.n_step", 1),
            ("model.use_noisy", False),
        ),
    ),
    Variant(
        key="dueling_ddqn",
        label="Dueling DDQN",
        color="#59A14F",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", False),
            ("agent.n_step", 1),
            ("model.use_noisy", False),
        ),
    ),
    Variant(
        key="d3qn_per_nstep",
        label="D3QN + PER + 3-step",
        color="#E15759",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", True),
            ("agent.n_step", 3),
            ("model.use_noisy", False),
        ),
    ),
    Variant(
        key="d3qn_per_nstep_noisy",
        label="D3QN + PER + 3-step + Noisy",
        color="#B07AA1",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", True),
            ("agent.n_step", 3),
            ("model.use_noisy", True),
        ),
    ),
)

NOISY_TUNING_VARIANTS: tuple[Variant, ...] = (
    Variant(
        key="dueling_ddqn",
        label="Dueling DDQN",
        color="#59A14F",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", False),
            ("agent.n_step", 1),
            ("model.use_noisy", False),
        ),
    ),
    Variant(
        key="dueling_noisy_full_sigma05",
        label="Noisy Full (sigma=0.5)",
        color="#E15759",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", False),
            ("agent.n_step", 1),
            ("model.use_noisy", True),
            ("model.noisy_sigma_init", 0.5),
            ("model.noisy_head_only", False),
        ),
    ),
    Variant(
        key="dueling_noisy_head_sigma01",
        label="Noisy Head (sigma=0.1)",
        color="#B07AA1",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", False),
            ("agent.n_step", 1),
            ("agent.batch_size", 32),
            ("agent.learning_rate", 0.00025),
            ("agent.target_update", 10),
            ("model.use_noisy", True),
            ("model.noisy_sigma_init", 0.1),
            ("model.noisy_head_only", True),
        ),
    ),
    Variant(
        key="dueling_noisy_head_sigma02",
        label="Noisy Head (sigma=0.2)",
        color="#4E79A7",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", False),
            ("agent.n_step", 1),
            ("agent.batch_size", 32),
            ("agent.learning_rate", 0.00025),
            ("agent.target_update", 10),
            ("model.use_noisy", True),
            ("model.noisy_sigma_init", 0.2),
            ("model.noisy_head_only", True),
        ),
    ),
)

NOISY_FULL_FOLLOWUP_VARIANTS: tuple[Variant, ...] = (
    Variant(
        key="dueling_ddqn",
        label="Dueling DDQN",
        color="#59A14F",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", False),
            ("agent.n_step", 1),
            ("model.use_noisy", False),
        ),
    ),
    Variant(
        key="dueling_noisy_full_sigma05",
        label="Noisy Full (sigma=0.5, lr=0.0005)",
        color="#E15759",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", False),
            ("agent.n_step", 1),
            ("model.use_noisy", True),
            ("model.noisy_sigma_init", 0.5),
            ("model.noisy_head_only", False),
        ),
    ),
    Variant(
        key="dueling_noisy_full_sigma05_lr25",
        label="Noisy Full (sigma=0.5, lr=0.00025)",
        color="#4E79A7",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", False),
            ("agent.n_step", 1),
            ("agent.batch_size", 32),
            ("agent.learning_rate", 0.00025),
            ("agent.target_update", 10),
            ("model.use_noisy", True),
            ("model.noisy_sigma_init", 0.5),
            ("model.noisy_head_only", False),
        ),
    ),
    Variant(
        key="dueling_noisy_full_sigma03_lr25",
        label="Noisy Full (sigma=0.3, lr=0.00025)",
        color="#B07AA1",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", False),
            ("agent.n_step", 1),
            ("agent.batch_size", 32),
            ("agent.learning_rate", 0.00025),
            ("agent.target_update", 10),
            ("model.use_noisy", True),
            ("model.noisy_sigma_init", 0.3),
            ("model.noisy_head_only", False),
        ),
    ),
)

FINAL_CONFIRMATION_VARIANTS: tuple[Variant, ...] = (
    Variant(
        key="dueling_ddqn",
        label="Dueling DDQN",
        color="#59A14F",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", False),
            ("agent.n_step", 1),
            ("model.use_noisy", False),
        ),
    ),
    Variant(
        key="d3qn_per_nstep",
        label="D3QN + PER + 3-step",
        color="#E15759",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", True),
            ("agent.n_step", 3),
            ("model.use_noisy", False),
        ),
    ),
    Variant(
        key="dueling_noisy_full_sigma05",
        label="Noisy Full (sigma=0.5, lr=0.0005)",
        color="#F28E2B",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", False),
            ("agent.n_step", 1),
            ("model.use_noisy", True),
            ("model.noisy_sigma_init", 0.5),
            ("model.noisy_head_only", False),
        ),
    ),
    Variant(
        key="dueling_noisy_full_sigma03_lr25",
        label="Noisy Full (sigma=0.3, lr=0.00025)",
        color="#4E79A7",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", False),
            ("agent.n_step", 1),
            ("agent.batch_size", 32),
            ("agent.learning_rate", 0.00025),
            ("agent.target_update", 10),
            ("model.use_noisy", True),
            ("model.noisy_sigma_init", 0.3),
            ("model.noisy_head_only", False),
        ),
    ),
)

OBSERVATION_UPGRADE_VARIANTS: tuple[Variant, ...] = (
    Variant(
        key="dueling_ddqn_raw",
        label="Dueling DDQN (raw)",
        color="#59A14F",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", False),
            ("agent.n_step", 1),
            ("model.use_noisy", False),
            ("env.observation_mode", "raw"),
            ("env.include_account_features", False),
        ),
    ),
    Variant(
        key="dueling_ddqn_returns_account",
        label="Dueling DDQN (returns + account)",
        color="#8CD17D",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", False),
            ("agent.n_step", 1),
            ("model.use_noisy", False),
            ("env.observation_mode", "returns"),
            ("env.include_account_features", True),
        ),
    ),
    Variant(
        key="d3qn_per_nstep_raw",
        label="D3QN + PER + 3-step (raw)",
        color="#E15759",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", True),
            ("agent.n_step", 3),
            ("model.use_noisy", False),
            ("env.observation_mode", "raw"),
            ("env.include_account_features", False),
        ),
    ),
    Variant(
        key="d3qn_per_nstep_returns_account",
        label="D3QN + PER + 3-step (returns + account)",
        color="#FF9D9A",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", True),
            ("agent.n_step", 3),
            ("model.use_noisy", False),
            ("env.observation_mode", "returns"),
            ("env.include_account_features", True),
        ),
    ),
    Variant(
        key="dueling_noisy_full_sigma05_raw",
        label="Noisy Full (raw)",
        color="#4E79A7",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", False),
            ("agent.n_step", 1),
            ("model.use_noisy", True),
            ("model.noisy_sigma_init", 0.5),
            ("model.noisy_head_only", False),
            ("env.observation_mode", "raw"),
            ("env.include_account_features", False),
        ),
    ),
    Variant(
        key="dueling_noisy_full_sigma05_returns_account",
        label="Noisy Full (returns + account)",
        color="#A0CBE8",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", False),
            ("agent.n_step", 1),
            ("model.use_noisy", True),
            ("model.noisy_sigma_init", 0.5),
            ("model.noisy_head_only", False),
            ("env.observation_mode", "returns"),
            ("env.include_account_features", True),
        ),
    ),
)

FEATURE_ONLY_VARIANTS: tuple[Variant, ...] = (
    Variant(
        key="dqn_returns_account",
        label="DQN",
        color="#4E79A7",
        overrides=(
            ("model.type", "mlp"),
            ("agent.double", False),
            ("agent.per_enabled", False),
            ("agent.n_step", 1),
            ("model.use_noisy", False),
            ("env.observation_mode", "returns"),
            ("env.include_account_features", True),
        ),
    ),
    Variant(
        key="ddqn_returns_account",
        label="DDQN",
        color="#F28E2B",
        overrides=(
            ("model.type", "mlp"),
            ("agent.double", True),
            ("agent.per_enabled", False),
            ("agent.n_step", 1),
            ("model.use_noisy", False),
            ("env.observation_mode", "returns"),
            ("env.include_account_features", True),
        ),
    ),
    Variant(
        key="dueling_ddqn_returns_account",
        label="Duel DDQN",
        color="#59A14F",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", False),
            ("agent.n_step", 1),
            ("model.use_noisy", False),
            ("env.observation_mode", "returns"),
            ("env.include_account_features", True),
        ),
    ),
    Variant(
        key="d3qn_per_nstep_returns_account",
        label="D3QN (+PER+3-step)",
        color="#FF9D9A",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", True),
            ("agent.n_step", 3),
            ("model.use_noisy", False),
            ("env.observation_mode", "returns"),
            ("env.include_account_features", True),
        ),
    ),
    Variant(
        key="d3qn_per_nstep_noisy_returns_account",
        label="D3QN+Noisy",
        color="#A0CBE8",
        overrides=(
            ("model.type", "mlp_dueling"),
            ("agent.double", True),
            ("agent.per_enabled", True),
            ("agent.n_step", 3),
            ("model.use_noisy", True),
            ("model.noisy_sigma_init", 0.5),
            ("model.noisy_head_only", False),
            ("env.observation_mode", "returns"),
            ("env.include_account_features", True),
        ),
    ),
)

VARIANT_SETS: dict[str, tuple[Variant, ...]] = {
    "default": DEFAULT_VARIANTS,
    "noisy_tuning": NOISY_TUNING_VARIANTS,
    "noisy_full_followup": NOISY_FULL_FOLLOWUP_VARIANTS,
    "final_confirmation": FINAL_CONFIRMATION_VARIANTS,
    "observation_upgrade": OBSERVATION_UPGRADE_VARIANTS,
    "feature_only": FEATURE_ONLY_VARIANTS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AAPL DQN ablation experiments and export figures.")
    parser.add_argument("--config", type=str, default="configs/aapl_baseline.yaml")
    parser.add_argument("--output-dir", type=str, default="results/aapl_dqn_ablation")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--start-date", type=str, default="2010-01-01")
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--trading-period", type=int, default=252)
    parser.add_argument("--train-split", type=float, default=1.0)
    parser.add_argument("--num-episodes", type=int, default=12)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42])
    parser.add_argument("--variant-set", type=str, choices=sorted(VARIANT_SETS), default="default")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _set_nested_attr(config: Config, path: str, value: object) -> None:
    current = config
    parts = path.split(".")
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], value)


def _prepare_config(base_config: Config, args: argparse.Namespace, seed: int, variant: Variant) -> Config:
    config = copy.deepcopy(base_config)
    config.data.start_date = args.start_date
    config.data.end_date = args.end_date
    config.env.trading_period = args.trading_period
    config.env.train_split = args.train_split
    config.train.num_episodes = args.num_episodes
    config.train.max_steps_per_episode = None
    config.train.max_total_steps = None
    config.train.log_interval = 1
    config.train.checkpoint_interval = args.num_episodes
    config.train.eval_interval = 0
    config.train.eval_episodes = args.eval_episodes
    config.train.resample_train_window_each_episode = False
    config.eval.num_episodes = args.eval_episodes
    config.eval.epsilon = 0.0
    config.eval.fixed_windows = True
    config.eval.fixed_windows_seed = 20240101
    config.eval.seed = 20240101
    config.run.seed = seed
    config.run.device = args.device
    for path, value in variant.overrides:
        _set_nested_attr(config, path, value)
    return config


def _build_run_paths(output_root: Path, variant: Variant, seed: int) -> RunPaths:
    run_dir = output_root / "runs" / variant.key / f"seed_{seed}"
    return RunPaths(
        run_dir=run_dir,
        checkpoints_dir=run_dir / "checkpoints",
        metrics_csv=run_dir / "metrics.csv",
        tensorboard_dir=run_dir / "tensorboard",
        config_resolved=run_dir / "config_resolved.yaml",
    )


def _load_training_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "episode" in df.columns:
        df["episode"] = df["episode"].astype(int) + 1
    return df


def _write_eval_outputs(run_dir: Path, checkpoint_path: Path, metrics: dict[str, float], mean_return: float) -> None:
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "eval_summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "mean_reward_return": mean_return,
                "metrics": metrics,
            },
            indent=2,
        )
    )
    with (eval_dir / "eval_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return mean(values) if values else 0.0


def _std(values: Iterable[float]) -> float:
    values = list(values)
    return pstdev(values) if len(values) > 1 else 0.0


def _scale_linear(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if math.isclose(src_min, src_max):
        return (dst_min + dst_max) / 2.0
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FAFAFA"/>',
        '<style>',
        "text { font-family: Helvetica, Arial, sans-serif; fill: #1F2937; }",
        ".title { font-size: 24px; font-weight: 700; }",
        ".axis { font-size: 12px; }",
        ".label { font-size: 13px; font-weight: 600; }",
        ".small { font-size: 11px; fill: #4B5563; }",
        "</style>",
    ]


def _write_svg(path: Path, lines: list[str]) -> None:
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_bar_chart_svg(
    path: Path,
    title: str,
    y_label: str,
    labels: list[str],
    values: list[float],
    errors: list[float],
    colors: list[str],
) -> None:
    width, height = 980, 560
    margin_left, margin_right, margin_top, margin_bottom = 90, 30, 70, 120
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    y_min = min(0.0, min(v - e for v, e in zip(values, errors)))
    y_max = max(0.0, max(v + e for v, e in zip(values, errors)))
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0
    pad = max((y_max - y_min) * 0.1, 1e-6)
    y_min -= pad
    y_max += pad
    zero_y = _scale_linear(0.0, y_min, y_max, margin_top + plot_height, margin_top)

    lines = _svg_header(width, height)
    lines.append(f'<text x="{margin_left}" y="36" class="title">{title}</text>')
    lines.append(
        f'<text x="{margin_left}" y="56" class="small">Bars show mean across seeds; whiskers show one standard deviation.</text>'
    )
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#374151" stroke-width="1.5"/>'
    )
    lines.append(
        f'<line x1="{margin_left}" y1="{zero_y:.2f}" x2="{margin_left + plot_width}" y2="{zero_y:.2f}" stroke="#9CA3AF" stroke-width="1.2"/>'
    )

    for tick_idx in range(6):
        tick_value = y_min + (y_max - y_min) * tick_idx / 5.0
        tick_y = _scale_linear(tick_value, y_min, y_max, margin_top + plot_height, margin_top)
        lines.append(
            f'<line x1="{margin_left - 6}" y1="{tick_y:.2f}" x2="{margin_left + plot_width}" y2="{tick_y:.2f}" stroke="#E5E7EB" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{margin_left - 10}" y="{tick_y + 4:.2f}" text-anchor="end" class="axis">{tick_value:.2f}</text>'
        )

    slot_width = plot_width / max(len(labels), 1)
    bar_width = min(90.0, slot_width * 0.58)
    for idx, (label, value, error, color) in enumerate(zip(labels, values, errors, colors)):
        center_x = margin_left + slot_width * (idx + 0.5)
        y_value = _scale_linear(value, y_min, y_max, margin_top + plot_height, margin_top)
        rect_y = min(y_value, zero_y)
        rect_height = max(abs(zero_y - y_value), 1.0)
        lines.append(
            f'<rect x="{center_x - bar_width / 2:.2f}" y="{rect_y:.2f}" width="{bar_width:.2f}" height="{rect_height:.2f}" fill="{color}" rx="6"/>'
        )
        err_top = _scale_linear(value + error, y_min, y_max, margin_top + plot_height, margin_top)
        err_bottom = _scale_linear(value - error, y_min, y_max, margin_top + plot_height, margin_top)
        lines.append(
            f'<line x1="{center_x:.2f}" y1="{err_top:.2f}" x2="{center_x:.2f}" y2="{err_bottom:.2f}" stroke="#111827" stroke-width="1.5"/>'
        )
        lines.append(
            f'<line x1="{center_x - 10:.2f}" y1="{err_top:.2f}" x2="{center_x + 10:.2f}" y2="{err_top:.2f}" stroke="#111827" stroke-width="1.5"/>'
        )
        lines.append(
            f'<line x1="{center_x - 10:.2f}" y1="{err_bottom:.2f}" x2="{center_x + 10:.2f}" y2="{err_bottom:.2f}" stroke="#111827" stroke-width="1.5"/>'
        )
        lines.append(
            f'<text x="{center_x:.2f}" y="{margin_top + plot_height + 22}" text-anchor="middle" class="label">{label}</text>'
        )
        lines.append(
            f'<text x="{center_x:.2f}" y="{rect_y - 8:.2f}" text-anchor="middle" class="small">{value:.3f}</text>'
        )

    lines.append(
        f'<text x="24" y="{margin_top + plot_height / 2:.2f}" transform="rotate(-90 24,{margin_top + plot_height / 2:.2f})" class="label">{y_label}</text>'
    )
    _write_svg(path, lines)


def _build_line_chart_svg(
    path: Path,
    title: str,
    y_label: str,
    x_label: str,
    subtitle: str,
    curve_df: pd.DataFrame,
    variants: tuple[Variant, ...],
    x_column: str,
    y_column: str,
) -> None:
    width, height = 980, 560
    margin_left, margin_right, margin_top, margin_bottom = 90, 30, 70, 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    x_min = float(curve_df[x_column].min())
    x_max = float(curve_df[x_column].max())
    y_min = float(curve_df[y_column].min())
    y_max = float(curve_df[y_column].max())
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0
    pad = max((y_max - y_min) * 0.1, 1e-6)
    y_min -= pad
    y_max += pad

    lines = _svg_header(width, height)
    lines.append(f'<text x="{margin_left}" y="36" class="title">{title}</text>')
    lines.append(f'<text x="{margin_left}" y="56" class="small">{subtitle}</text>')
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#374151" stroke-width="1.5"/>'
    )
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#374151" stroke-width="1.5"/>'
    )

    x_tick_count = min(10, max(2, int(x_max - x_min) + 1))
    for tick_idx in range(x_tick_count):
        tick_value = x_min + (x_max - x_min) * tick_idx / max(x_tick_count - 1, 1)
        x = _scale_linear(float(tick_value), x_min, x_max, margin_left, margin_left + plot_width)
        lines.append(
            f'<line x1="{x:.2f}" y1="{margin_top + plot_height}" x2="{x:.2f}" y2="{margin_top + plot_height + 6}" stroke="#374151" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{margin_top + plot_height + 24}" text-anchor="middle" class="axis">{int(round(tick_value))}</text>'
        )
    for tick_idx in range(6):
        value = y_min + (y_max - y_min) * tick_idx / 5.0
        y = _scale_linear(value, y_min, y_max, margin_top + plot_height, margin_top)
        lines.append(
            f'<line x1="{margin_left - 6}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#E5E7EB" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" class="axis">{value:.2f}</text>'
        )

    legend_x = margin_left + plot_width - 220
    legend_y = margin_top + 20
    for idx, variant in enumerate(variants):
        variant_df = curve_df[curve_df["variant"] == variant.key].sort_values(x_column)
        points = []
        for row in variant_df.itertuples():
            x = _scale_linear(float(getattr(row, x_column)), x_min, x_max, margin_left, margin_left + plot_width)
            y = _scale_linear(float(getattr(row, y_column)), y_min, y_max, margin_top + plot_height, margin_top)
            points.append(f"{x:.2f},{y:.2f}")
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{variant.color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        item_y = legend_y + idx * 24
        lines.append(
            f'<line x1="{legend_x}" y1="{item_y}" x2="{legend_x + 18}" y2="{item_y}" stroke="{variant.color}" stroke-width="4" stroke-linecap="round"/>'
        )
        lines.append(
            f'<text x="{legend_x + 26}" y="{item_y + 4}" class="axis">{variant.label}</text>'
        )

    lines.append(
        f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 18}" text-anchor="middle" class="label">{x_label}</text>'
    )
    lines.append(
        f'<text x="24" y="{margin_top + plot_height / 2:.2f}" transform="rotate(-90 24,{margin_top + plot_height / 2:.2f})" class="label">{y_label}</text>'
    )
    _write_svg(path, lines)


def _write_markdown_report(
    output_root: Path,
    summary_df: pd.DataFrame,
    args: argparse.Namespace,
    variants: tuple[Variant, ...],
) -> None:
    report_path = output_root / "ablation_report.md"
    best_return = summary_df.sort_values("mean_return_rate_mean", ascending=False).iloc[0]
    best_sharpe = summary_df.sort_values("sharpe_ratio_mean", ascending=False).iloc[0]
    if args.variant_set == "noisy_tuning":
        report_title = "AAPL Noisy Tuning"
    elif args.variant_set == "noisy_full_followup":
        report_title = "AAPL Noisy Full Follow-Up"
    elif args.variant_set == "final_confirmation":
        report_title = "AAPL Final Confirmation"
    elif args.variant_set == "observation_upgrade":
        report_title = "AAPL Observation Upgrade"
    elif args.variant_set == "feature_only":
        report_title = "AAPL Feature-Only Comparison"
    else:
        report_title = "AAPL DQN Ablation"
    lines = [
        f"# {report_title}",
        "",
        "## Setup",
        "",
        f"- Data range: {args.start_date} to {args.end_date or 'latest available row'}",
        f"- Trading period per episode: {args.trading_period} steps",
        f"- Training episodes per run: {args.num_episodes}",
        f"- Evaluation episodes per run: {args.eval_episodes}",
        f"- Seeds: {', '.join(str(seed) for seed in args.seeds)}",
        f"- Variants: {', '.join(variant.label for variant in variants)}",
        "",
        "## Best Variants",
        "",
        f"- Highest mean return rate: **{best_return['label']}** ({best_return['mean_return_rate_mean'] * 100:.2f}% +/- {best_return['mean_return_rate_std'] * 100:.2f}%)",
        f"- Highest mean Sharpe ratio: **{best_sharpe['label']}** ({best_sharpe['sharpe_ratio_mean']:.3f} +/- {best_sharpe['sharpe_ratio_std']:.3f})",
        "",
        "## Files",
        "",
        "- `ablation_runs.csv`: one row per variant-seed run",
        "- `ablation_summary.csv`: aggregated mean and standard deviation by variant",
        "- `training_curve_summary.csv`: mean episode reward curves across seeds",
        "- `eval_curve_summary.csv`: mean evaluation cumulative return curves across seeds and windows",
        "- `figures/`: SVG charts ready to use in the report",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    variants = VARIANT_SETS[args.variant_set]
    output_root = Path(args.output_dir)
    figures_dir = output_root / "figures"
    output_root.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_run_logger("ablation", output_root, log_filename="ablation.log")

    base_config = load_config(Path(args.config))
    run_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    eval_curve_rows: list[dict[str, object]] = []

    for variant in variants:
        for seed in args.seeds:
            config = _prepare_config(base_config, args, seed, variant)
            run_paths = _build_run_paths(output_root, variant, seed)
            checkpoint_path = run_paths.checkpoints_dir / "checkpoint_latest.pt"
            metrics_path = run_paths.metrics_csv

            if args.force or not checkpoint_path.exists() or not metrics_path.exists():
                logger.info("Training %s with seed %s", variant.label, seed)
                train(config, run_paths)
            else:
                logger.info("Skipping existing run for %s seed %s", variant.label, seed)
                save_config(config, run_paths.config_resolved)

            mean_reward_return, eval_metrics, eval_cumulative_returns = evaluate(
                config=config,
                checkpoint_path=checkpoint_path,
                episodes=config.eval.num_episodes,
                epsilon=config.eval.epsilon,
                device=config.run.device,
                eval_seed=config.eval.seed,
            )
            _write_eval_outputs(run_paths.run_dir, checkpoint_path, eval_metrics, mean_reward_return)

            training_df = _load_training_metrics(metrics_path)
            for row in training_df.itertuples():
                curve_rows.append(
                    {
                        "variant": variant.key,
                        "label": variant.label,
                        "seed": seed,
                        "episode": int(row.episode),
                        "reward_return": float(row.reward_return),
                        "avg_loss": float(row.avg_loss),
                        "avg_q": float(row.avg_q),
                    }
                )
            for episode_index, episode_curve in enumerate(eval_cumulative_returns):
                for step_index, cumulative_return in enumerate(episode_curve):
                    eval_curve_rows.append(
                        {
                            "variant": variant.key,
                            "label": variant.label,
                            "seed": seed,
                            "eval_episode": episode_index + 1,
                            "step": step_index,
                            "cumulative_return": float(cumulative_return),
                        }
                    )

            run_rows.append(
                {
                    "variant": variant.key,
                    "label": variant.label,
                    "seed": seed,
                    "color": variant.color,
                    "mean_reward_return": mean_reward_return,
                    **eval_metrics,
                }
            )
            logger.info(
                "Finished %s seed %s | mean_return_rate %.4f | sharpe %.4f",
                variant.label,
                seed,
                eval_metrics["mean_return_rate"],
                eval_metrics["sharpe_ratio"],
            )

    runs_df = pd.DataFrame(run_rows)
    curves_df = pd.DataFrame(curve_rows)
    runs_df.to_csv(output_root / "ablation_runs.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    for variant in variants:
        variant_df = runs_df[runs_df["variant"] == variant.key]
        summary_rows.append(
            {
                "variant": variant.key,
                "label": variant.label,
                "color": variant.color,
                "seeds": len(variant_df),
                "mean_reward_return_mean": _mean(variant_df["mean_reward_return"]),
                "mean_reward_return_std": _std(variant_df["mean_reward_return"]),
                "mean_return_rate_mean": _mean(variant_df["mean_return_rate"]),
                "mean_return_rate_std": _std(variant_df["mean_return_rate"]),
                "sharpe_ratio_mean": _mean(variant_df["sharpe_ratio"]),
                "sharpe_ratio_std": _std(variant_df["sharpe_ratio"]),
                "annualized_sharpe_ratio_mean": _mean(variant_df["annualized_sharpe_ratio"]),
                "annualized_sharpe_ratio_std": _std(variant_df["annualized_sharpe_ratio"]),
                "max_drawdown_mean": _mean(variant_df["max_drawdown"]),
                "max_drawdown_std": _std(variant_df["max_drawdown"]),
                "win_rate_mean": _mean(variant_df["win_rate"]),
                "win_rate_std": _std(variant_df["win_rate"]),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "ablation_summary.csv", index=False)

    curve_summary_df = (
        curves_df.groupby(["variant", "label", "episode"], as_index=False)["reward_return"]
        .mean()
        .rename(columns={"reward_return": "mean_reward_return"})
    )
    curve_summary_df.to_csv(output_root / "training_curve_summary.csv", index=False)
    eval_curves_df = pd.DataFrame(eval_curve_rows)
    eval_curve_summary_df = (
        eval_curves_df.groupby(["variant", "label", "step"], as_index=False)["cumulative_return"]
        .mean()
        .rename(columns={"cumulative_return": "mean_cumulative_return"})
    )
    eval_curve_summary_df.to_csv(output_root / "eval_curve_summary.csv", index=False)

    labels = summary_df["label"].tolist()
    colors = summary_df["color"].tolist()
    _build_bar_chart_svg(
        figures_dir / "mean_return_rate.svg",
        title="AAPL Ablation: Mean Return Rate",
        y_label="Mean Return Rate",
        labels=labels,
        values=summary_df["mean_return_rate_mean"].tolist(),
        errors=summary_df["mean_return_rate_std"].tolist(),
        colors=colors,
    )
    _build_bar_chart_svg(
        figures_dir / "mean_sharpe_ratio.svg",
        title="AAPL Ablation: Mean Sharpe Ratio",
        y_label="Mean Sharpe Ratio",
        labels=labels,
        values=summary_df["sharpe_ratio_mean"].tolist(),
        errors=summary_df["sharpe_ratio_std"].tolist(),
        colors=colors,
    )
    _build_line_chart_svg(
        figures_dir / "training_reward_curve.svg",
        title="AAPL Ablation: Training Reward Curve",
        y_label="Episode Reward",
        x_label="Episode",
        subtitle="Curves show mean training episode reward across seeds.",
        curve_df=curve_summary_df,
        variants=variants,
        x_column="episode",
        y_column="mean_reward_return",
    )
    _build_line_chart_svg(
        figures_dir / "mean_cumulative_return_curve.svg",
        title="AAPL Ablation: Mean Cumulative Return Curve",
        y_label="Mean Cumulative Return",
        x_label="Evaluation Step",
        subtitle="Curves show mean evaluation cumulative return across seeds and fixed windows.",
        curve_df=eval_curve_summary_df,
        variants=variants,
        x_column="step",
        y_column="mean_cumulative_return",
    )
    _write_markdown_report(output_root, summary_df, args, variants)
    logger.info("Ablation results written to %s", output_root)


if __name__ == "__main__":
    main()
