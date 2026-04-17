from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_q_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(predicted, target)


def weighted_huber_q_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    per_item = F.smooth_l1_loss(predicted, target, reduction="none")
    return (per_item * weights).mean()
