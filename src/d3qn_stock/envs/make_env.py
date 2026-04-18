from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from d3qn_stock.envs.trading_env_discrete_capital import DiscreteCapitalTradingEnvironment


def _coerce_numeric_column(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def load_price_data(
    path: Path,
    close_column: str = "Close",
    date_column: str = "Date",
    open_column: Optional[str] = "Open",
    high_column: Optional[str] = "High",
    low_column: Optional[str] = "Low",
    volume_column: Optional[str] = "Volume",
) -> pd.DataFrame:
    df = pd.read_csv(path)

    rename_map: dict[str, str] = {}
    column_map = {
        close_column: "Close",
        date_column: "Date",
    }
    if open_column:
        column_map[open_column] = "Open"
    if high_column:
        column_map[high_column] = "High"
    if low_column:
        column_map[low_column] = "Low"
    if volume_column:
        column_map[volume_column] = "Volume"

    for source, target in column_map.items():
        if source in df.columns and source != target:
            rename_map[source] = target
    df = df.rename(columns=rename_map)

    required_columns = {"Date", "Close"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}")

    for column in ("Open", "High", "Low", "Close", "Volume"):
        if column in df.columns:
            df[column] = _coerce_numeric_column(df[column])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date").reset_index(drop=True)
    return df


def filter_date_range(
    df: pd.DataFrame,
    date_column: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    if start_date is None and end_date is None:
        return df
    mask = pd.Series(True, index=df.index)
    if start_date is not None:
        mask &= df[date_column] >= start_date
    if end_date is not None:
        mask &= df[date_column] <= end_date
    return df.loc[mask].reset_index(drop=True)


def sample_train_test_split(
    df: pd.DataFrame,
    trading_period: int,
    train_split: float,
    index: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if trading_period <= 0:
        raise ValueError("trading_period must be > 0.")
    if len(df) <= trading_period + 1:
        raise ValueError("Dataframe is too short for the requested trading period.")
    if index is None:
        import random

        index = random.randrange(len(df) - trading_period - 1)
    train_size = max(1, int(trading_period * train_split))
    train_df = df[index : index + train_size]
    test_df = df[index + train_size : index + trading_period]
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_env(
    df: pd.DataFrame,
    reward: str,
    window_size: int,
    device: str,
    observation_mode: str = "raw",
    include_account_features: bool = False,
    trading_period: Optional[int] = None,
    max_positions: Optional[int] = None,
    max_exposure_ratio: Optional[float] = 1.0,
    sell_mode: str = "all",
    buy_fractions: Optional[list[float]] = None,
    sell_fractions: Optional[list[float]] = None,
    action_number: Optional[int] = None,
    initial_capital: float = 100_000.0,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 2.0,
    invalid_sell_penalty: float = 0.1,
    blocked_trade_penalty: float = 0.0,
    min_hold_steps: int = 0,
    trade_cooldown_steps: int = 0,
    dynamic_exposure_enabled: bool = False,
    dynamic_exposure_vol_window: int = 30,
    dynamic_exposure_min_scale: float = 0.5,
    dynamic_exposure_strength: float = 1.0,
    min_equity_ratio: float = 0.2,
    stop_on_bankruptcy: bool = True,
    sr_window: int = 20,
    sr_clip: float = 1.0,
    periods_per_year: float = 252.0,
) -> DiscreteCapitalTradingEnvironment:
    env = DiscreteCapitalTradingEnvironment(
        df,
        reward=reward,
        window_size=window_size,
        observation_mode=observation_mode,
        include_account_features=include_account_features,
        trading_period=trading_period,
        max_positions=max_positions,
        max_exposure_ratio=max_exposure_ratio,
        sell_mode=sell_mode,
        buy_fractions=buy_fractions,
        sell_fractions=sell_fractions,
        action_number=action_number,
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        invalid_sell_penalty=invalid_sell_penalty,
        blocked_trade_penalty=blocked_trade_penalty,
        min_hold_steps=min_hold_steps,
        trade_cooldown_steps=trade_cooldown_steps,
        dynamic_exposure_enabled=dynamic_exposure_enabled,
        dynamic_exposure_vol_window=dynamic_exposure_vol_window,
        dynamic_exposure_min_scale=dynamic_exposure_min_scale,
        dynamic_exposure_strength=dynamic_exposure_strength,
        min_equity_ratio=min_equity_ratio,
        stop_on_bankruptcy=stop_on_bankruptcy,
        sr_window=sr_window,
        sr_clip=sr_clip,
        periods_per_year=periods_per_year,
        annualize_sr_reward=False,
        device=device,
    )
    return env
