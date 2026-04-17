from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from d3qn_stock.algos.d3qn.trainer import Config, load_config, save_config, train
from d3qn_stock.utils.path import build_run_name, build_run_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a D3QN stock trading agent.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--reward", type=str, default=None, choices=["profit", "sr", "sr_enhanced"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--trading-period", type=int, default=None)
    parser.add_argument("--train-split", type=float, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values, for example --override agent.learning_rate=0.00025",
    )
    return parser.parse_args()


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    if args.seed is not None:
        config.run.seed = args.seed
    if args.num_episodes is not None:
        config.train.num_episodes = args.num_episodes
    if args.max_steps_per_episode is not None:
        config.train.max_steps_per_episode = args.max_steps_per_episode
    if args.total_steps is not None:
        config.train.max_total_steps = args.total_steps
    if args.reward is not None:
        config.env.reward = args.reward
    if args.device is not None:
        config.run.device = args.device
    if args.trading_period is not None:
        config.env.trading_period = args.trading_period
    if args.train_split is not None:
        config.env.train_split = args.train_split
    if args.log_interval is not None:
        config.train.log_interval = args.log_interval
    if args.checkpoint_interval is not None:
        config.train.checkpoint_interval = args.checkpoint_interval
    return config


def _set_config_value(config: Config, path: str, value) -> None:
    parts = path.split(".")
    current = config
    for part in parts[:-1]:
        if not hasattr(current, part):
            raise ValueError(f"Unknown config field: {path}")
        current = getattr(current, part)
    final = parts[-1]
    if not hasattr(current, final):
        raise ValueError(f"Unknown config field: {path}")
    setattr(current, final, value)


def apply_kv_overrides(config: Config, overrides: list[str]) -> Config:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override format: {item}")
        key, raw_value = item.split("=", 1)
        _set_config_value(config, key, yaml.safe_load(raw_value))
    return config


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    config = apply_overrides(config, args)
    config = apply_kv_overrides(config, args.override)

    if args.run_name is None:
        run_name = build_run_name()
    else:
        run_name = f"{args.run_name}_{secrets.token_hex(3)}"
    run_paths = build_run_paths(Path(args.output_dir), run_name)
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_paths.config_resolved)
    train(config, run_paths)


if __name__ == "__main__":
    main()
