from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RunPaths:
    run_dir: Path
    checkpoints_dir: Path
    metrics_csv: Path
    tensorboard_dir: Path
    config_resolved: Path


def build_run_name(base_name: str | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base_name:
        return f"{base_name}_{timestamp}"
    return f"stock_d3qn_{timestamp}"


def build_run_paths(output_dir: Path, run_name: str) -> RunPaths:
    run_dir = output_dir / run_name
    checkpoints_dir = run_dir / "checkpoints"
    metrics_csv = run_dir / "metrics.csv"
    tensorboard_dir = run_dir / "tensorboard"
    config_resolved = run_dir / "config_resolved.yaml"
    return RunPaths(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        metrics_csv=metrics_csv,
        tensorboard_dir=tensorboard_dir,
        config_resolved=config_resolved,
    )
