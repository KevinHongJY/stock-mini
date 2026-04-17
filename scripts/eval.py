from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from d3qn_stock.algos.d3qn.trainer import config_from_dict, evaluate, load_config
from d3qn_stock.utils.checkpoint import load_checkpoint
from d3qn_stock.utils.logging import setup_run_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a D3QN stock trading agent.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--reward", type=str, default=None, choices=["profit", "sr", "sr_enhanced"])
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--trading-period", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def _resolve_output_dir(checkpoint_path: Path, explicit_output_dir: str | None) -> Path:
    if explicit_output_dir:
        return Path(explicit_output_dir)
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent / "eval"
    return checkpoint_path.parent / "eval"


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_dir = _resolve_output_dir(checkpoint_path, args.output_dir)
    logger = setup_run_logger("eval", output_dir, log_filename="eval.log")

    checkpoint = load_checkpoint(checkpoint_path, device=args.device or "auto")
    if "config" in checkpoint:
        config = config_from_dict(checkpoint["config"])
    else:
        config = load_config(Path(args.config))

    if args.device is not None:
        config.run.device = args.device
    if args.reward is not None:
        config.env.reward = args.reward
    if args.start_date is not None:
        config.data.start_date = args.start_date
    if args.end_date is not None:
        config.data.end_date = args.end_date
    if args.trading_period is not None:
        config.env.trading_period = args.trading_period

    episodes = args.episodes or config.eval.num_episodes
    epsilon = config.eval.epsilon if args.epsilon is None else args.epsilon
    mean_return, metrics, cumulative_returns = evaluate(
        config=config,
        checkpoint_path=checkpoint_path,
        episodes=episodes,
        epsilon=epsilon,
        device=config.run.device,
        eval_seed=config.eval.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "eval_summary.json"
    metrics_csv_path = output_dir / "eval_metrics.csv"
    cumulative_returns_path = output_dir / "cumulative_returns.json"

    summary = {
        "checkpoint": str(checkpoint_path),
        "episodes": episodes,
        "mean_reward_return": mean_return,
        "metrics": metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    cumulative_returns_path.write_text(json.dumps(cumulative_returns))
    with metrics_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])

    logger.info("Evaluation summary written to %s", summary_path)
    logger.info("Evaluation metrics written to %s", metrics_csv_path)


if __name__ == "__main__":
    main()
