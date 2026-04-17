from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def compute_sharpe_ratio(mean_return_rate: float, std_return_rate: float) -> float:
    if std_return_rate <= 1e-12:
        return 0.0
    return float(mean_return_rate / std_return_rate)


def compute_annualized_sharpe_ratio(
    mean_return_rate: float,
    std_return_rate: float,
    periods_per_year: float,
) -> float:
    raw_sharpe = compute_sharpe_ratio(mean_return_rate, std_return_rate)
    if raw_sharpe == 0.0 or periods_per_year <= 0.0:
        return 0.0
    return float(raw_sharpe * math.sqrt(periods_per_year))


def compute_max_drawdown_from_equity(equity_curve: Sequence[float]) -> float:
    if len(equity_curve) <= 1:
        return 0.0
    equity = np.asarray(equity_curve, dtype=float)
    if not np.isfinite(equity).all():
        return 0.0
    running_peak = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(running_peak, 1e-12) - 1.0
    return float(np.min(drawdown))


def compute_active_streak_lengths(values: Sequence[float], threshold: float = 1e-12) -> list[int]:
    streaks: list[int] = []
    current = 0
    for value in values:
        if float(value) > threshold:
            current += 1
        elif current > 0:
            streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    return streaks


def safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)
