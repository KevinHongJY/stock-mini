from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(
    path: Path,
    policy_state: Dict[str, Any],
    target_state: Dict[str, Any],
    optimizer_state: Dict[str, Any],
    config: Dict[str, Any],
    episode: int,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_state": policy_state,
        "target_state": target_state,
        "optimizer_state": optimizer_state,
        "config": config,
        "episode": episode,
        "step": step,
    }
    torch.save(payload, str(path))


def load_checkpoint(path: Path, device: str) -> Dict[str, Any]:
    if device == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved = device
    return torch.load(str(path), map_location=torch.device(resolved))
