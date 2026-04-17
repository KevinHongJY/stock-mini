from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch


class DiscreteCapitalTradingEnvironment:
    """Discrete-action trading environment with explicit cash and position accounting."""

    def __init__(
        self,
        data: pd.DataFrame,
        reward: str,
        window_size: int = 24,
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
        sr_window: int = 30,
        sr_clip: float = 1.0,
        periods_per_year: float = 252.0,
        annualize_sr_reward: bool = False,
        device: str = "auto",
    ) -> None:
        self.data = data
        valid_rewards = {"profit", "sr", "sr_enhanced"}
        if reward not in valid_rewards:
            raise ValueError(f"reward must be one of {sorted(valid_rewards)}")
        self.reward_f = reward
        self.window_size = int(window_size)
        self.trading_period = trading_period
        if max_positions is not None and max_positions < 1:
            raise ValueError("max_positions must be >= 1 or None.")
        valid_sell_modes = {"all", "one", "all_cap", "one_plus"}
        if sell_mode not in valid_sell_modes:
            raise ValueError("sell_mode must be 'all', 'one', 'all_cap', or 'one_plus'.")
        if sell_mode == "all_cap" and max_positions is None and max_exposure_ratio is None:
            raise ValueError("sell_mode='all_cap' requires max_positions or max_exposure_ratio.")
        self.max_positions = int(max_positions) if max_positions is not None else None
        if max_exposure_ratio is None:
            self.max_exposure_ratio = None
        else:
            self.max_exposure_ratio = float(max_exposure_ratio)
            if not (0.0 < self.max_exposure_ratio <= 1.0):
                raise ValueError("max_exposure_ratio must be in (0, 1].")
        self.sell_mode = sell_mode
        self.buy_fractions = self._normalize_fraction_list(buy_fractions, "buy_fractions")
        self.sell_fractions = self._normalize_fraction_list(sell_fractions, "sell_fractions")
        if self.buy_fractions and not self.sell_fractions:
            self.sell_fractions = [1.0]
        if self.sell_fractions and not self.buy_fractions:
            self.buy_fractions = [1.0]
        self.use_fractional_actions = bool(self.buy_fractions or self.sell_fractions)
        if self.use_fractional_actions and self.sell_mode not in {"all", "all_cap"}:
            raise ValueError("buy_fractions and sell_fractions require sell_mode='all' or 'all_cap'.")
        self.action_number = int(action_number) if action_number is not None else None
        if self.action_number is not None:
            if self.use_fractional_actions:
                expected_actions = 1 + len(self.buy_fractions) + len(self.sell_fractions)
                if self.action_number != expected_actions:
                    raise ValueError(
                        f"action_number={self.action_number} does not match fraction action space "
                        f"(expected {expected_actions})."
                    )
            elif self.action_number < 3:
                raise ValueError("action_number must be >= 3 when provided.")

        self.initial_capital = float(initial_capital)
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0.")
        self.transaction_cost_rate = float(transaction_cost_bps) / 10_000.0
        self.slippage_rate = float(slippage_bps) / 10_000.0
        if self.transaction_cost_rate < 0 or self.slippage_rate < 0:
            raise ValueError("transaction_cost_bps and slippage_bps must be >= 0.")
        self.invalid_sell_penalty = float(invalid_sell_penalty)
        if self.invalid_sell_penalty < 0:
            raise ValueError("invalid_sell_penalty must be >= 0.")
        self.blocked_trade_penalty = float(blocked_trade_penalty)
        if self.blocked_trade_penalty < 0:
            raise ValueError("blocked_trade_penalty must be >= 0.")
        self.min_hold_steps = int(min_hold_steps)
        if self.min_hold_steps < 0:
            raise ValueError("min_hold_steps must be >= 0.")
        self.trade_cooldown_steps = int(trade_cooldown_steps)
        if self.trade_cooldown_steps < 0:
            raise ValueError("trade_cooldown_steps must be >= 0.")
        self.dynamic_exposure_enabled = bool(dynamic_exposure_enabled and self.max_exposure_ratio is not None)
        self.dynamic_exposure_vol_window = int(dynamic_exposure_vol_window)
        if self.dynamic_exposure_vol_window < 2:
            raise ValueError("dynamic_exposure_vol_window must be >= 2.")
        self.dynamic_exposure_min_scale = float(dynamic_exposure_min_scale)
        if not (0.0 < self.dynamic_exposure_min_scale <= 1.0):
            raise ValueError("dynamic_exposure_min_scale must be in (0, 1].")
        self.dynamic_exposure_strength = float(dynamic_exposure_strength)
        if self.dynamic_exposure_strength < 0.0:
            raise ValueError("dynamic_exposure_strength must be >= 0.")
        self.min_equity_ratio = float(min_equity_ratio)
        if not (0.0 < self.min_equity_ratio <= 1.0):
            raise ValueError("min_equity_ratio must be in (0, 1].")
        self.stop_on_bankruptcy = bool(stop_on_bankruptcy)
        self.sr_window = int(sr_window)
        if self.sr_window < 2:
            raise ValueError("sr_window must be >= 2 for sample std (ddof=1).")
        self.sr_clip = float(sr_clip)
        if self.sr_clip < 0.0:
            raise ValueError("sr_clip must be >= 0.")
        self.periods_per_year = float(periods_per_year)
        if self.periods_per_year <= 0.0:
            raise ValueError("periods_per_year must be > 0.")
        self.annualize_sr_reward = bool(annualize_sr_reward)
        self.sr_scale = math.sqrt(self.periods_per_year) if self.annualize_sr_reward else 1.0
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )

        self.rng = np.random.default_rng()
        self.last_start_index: Optional[int] = None
        self.last_start_timestamp: Optional[str] = None
        self._episode_end_index: Optional[int] = None
        self.sr_eps = 1e-8
        close_values = np.asarray(self.data["Close"], dtype=float)
        self._rolling_vol = self._build_rolling_vol(close_values, self.dynamic_exposure_vol_window)
        valid_vol = self._rolling_vol[np.isfinite(self._rolling_vol) & (self._rolling_vol > self.sr_eps)]
        self.dynamic_exposure_ref_vol = float(np.median(valid_vol)) if valid_vol.size > 0 else 0.0
        self.reset(start_index=self.window_size - 1)

    @staticmethod
    def _normalize_fraction_list(values: Optional[list[float]], name: str) -> list[float]:
        if values is None:
            return []
        normalized: list[float] = []
        for value in values:
            fraction = float(value)
            if not (0.0 < fraction <= 1.0):
                raise ValueError(f"{name} values must be in (0, 1].")
            normalized.append(fraction)
        if not normalized:
            raise ValueError(f"{name} must not be an empty list when provided.")
        return normalized

    @staticmethod
    def _build_rolling_vol(close_values: np.ndarray, window: int) -> np.ndarray:
        clipped = np.clip(np.asarray(close_values, dtype=float), 1e-12, None)
        log_prices = np.log(clipped)
        log_returns = np.diff(log_prices, prepend=log_prices[0])
        vol = pd.Series(log_returns).rolling(window=window, min_periods=2).std(ddof=0).to_numpy(dtype=float)
        return np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

    def _effective_max_exposure_ratio(self) -> Optional[float]:
        if self.max_exposure_ratio is None:
            return None
        if not self.dynamic_exposure_enabled:
            return self.max_exposure_ratio
        index = min(max(self.t, 0), len(self._rolling_vol) - 1)
        current_vol = float(self._rolling_vol[index])
        ref_vol = float(self.dynamic_exposure_ref_vol)
        if current_vol <= self.sr_eps or ref_vol <= self.sr_eps:
            return self.max_exposure_ratio
        vol_ratio = current_vol / ref_vol
        if vol_ratio <= 1.0:
            scale = 1.0
        else:
            scale = 1.0 / (1.0 + self.dynamic_exposure_strength * (vol_ratio - 1.0))
        scale = float(np.clip(scale, self.dynamic_exposure_min_scale, 1.0))
        return float(self.max_exposure_ratio * scale)

    def _get_valid_start_range(self) -> Tuple[int, int]:
        min_start = self.window_size - 1
        max_start = len(self.data) - 1
        if self.trading_period is not None:
            max_start = len(self.data) - self.trading_period
        return min_start, max_start

    def _resolve_start_timestamp(self, start_index: int) -> Optional[str]:
        if "Date" not in self.data.columns:
            return None
        timestamp = self.data.iloc[start_index]["Date"]
        if isinstance(timestamp, pd.Timestamp):
            return timestamp.isoformat()
        return str(timestamp)

    def _current_price(self) -> float:
        index = min(max(self.t, 0), len(self.data) - 1)
        return float(self.data.iloc[index]["Close"])

    def _position_units(self) -> float:
        return float(sum(quantity for _, quantity in self.agent_positions))

    def _position_value(self, price: float) -> float:
        return float(self._position_units() * price)

    def _equity(self, price: float) -> float:
        return float(self.cash + self._position_value(price))

    def _max_buy_quantity(self, current_price: float) -> float:
        buy_price = current_price * (1.0 + self.slippage_rate)
        unit_fee = current_price * self.transaction_cost_rate
        unit_cash_cost = buy_price + unit_fee
        if unit_cash_cost <= self.sr_eps:
            return 0.0

        max_by_cash = max(0.0, self.cash / unit_cash_cost)
        max_by_exposure = float("inf")
        ratio = self._effective_max_exposure_ratio()
        if ratio is not None:
            position_qty = self._position_units()
            numerator = ratio * self.cash - (1.0 - ratio) * position_qty * current_price
            denominator = (1.0 - ratio) * current_price + ratio * unit_cash_cost
            if denominator <= self.sr_eps:
                max_by_exposure = 0.0
            else:
                max_by_exposure = max(0.0, numerator / denominator)

        max_by_positions = float("inf")
        if self.max_positions is not None:
            max_by_positions = max(0.0, float(self.max_positions) - self._position_units())

        return float(max(0.0, min(max_by_cash, max_by_exposure, max_by_positions)))

    def _execute_buy(self, current_price: float, quantity: float) -> float:
        qty = float(quantity)
        if qty <= self.sr_eps:
            return 0.0
        buy_price = current_price * (1.0 + self.slippage_rate)
        fee = qty * current_price * self.transaction_cost_rate
        required_cash = qty * buy_price + fee
        if required_cash > self.cash + 1e-12:
            affordable_qty = max(0.0, self.cash / (buy_price + current_price * self.transaction_cost_rate))
            qty = min(qty, affordable_qty)
            if qty <= self.sr_eps:
                return 0.0
            fee = qty * current_price * self.transaction_cost_rate
            required_cash = qty * buy_price + fee

        self.agent_positions.append((current_price, qty))
        self.cash -= required_cash
        self.total_turnover += qty * current_price
        self.total_fees += fee
        return qty

    def _execute_sell(self, current_price: float, quantity: float) -> float:
        qty_remaining = min(float(quantity), self._position_units())
        if qty_remaining <= self.sr_eps:
            return 0.0

        sold_lots: list[tuple[float, float]] = []
        while qty_remaining > self.sr_eps and self.agent_positions:
            entry, held_qty = self.agent_positions[0]
            sell_qty = min(held_qty, qty_remaining)
            sold_lots.append((entry, sell_qty))
            leftover_qty = held_qty - sell_qty
            if leftover_qty <= self.sr_eps:
                self.agent_positions.pop(0)
            else:
                self.agent_positions[0] = (entry, leftover_qty)
            qty_remaining -= sell_qty

        sold_quantity = float(sum(qty for _, qty in sold_lots))
        if sold_quantity <= self.sr_eps:
            return 0.0

        sell_price = current_price * (1.0 - self.slippage_rate)
        fee = sold_quantity * current_price * self.transaction_cost_rate
        proceeds = sold_quantity * sell_price - fee
        self.cash += proceeds
        self.total_turnover += sold_quantity * current_price
        self.total_fees += fee
        return float(sum((current_price - entry) * qty for entry, qty in sold_lots) - fee)

    def reset(
        self,
        seed: Optional[int] = None,
        start_index: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        min_start, max_start = self._get_valid_start_range()
        if max_start < min_start:
            raise ValueError(
                "Invalid start_index range "
                f"[{min_start}, {max_start}] for data length {len(self.data)} "
                f"and trading_period {self.trading_period}."
            )
        if start_index is None:
            start_index = int(self.rng.integers(min_start, max_start + 1))
        else:
            if not (min_start <= start_index <= max_start):
                raise ValueError(
                    "start_index must be in the range "
                    f"[{min_start}, {max_start}] but got {start_index}."
                )
            start_index = int(start_index)

        self.last_start_index = start_index
        self.last_start_timestamp = self._resolve_start_timestamp(start_index)
        self.t = start_index
        self.done = False
        if self.trading_period is not None:
            self._episode_end_index = start_index + self.trading_period - 1
        else:
            self._episode_end_index = len(self.data) - 1

        self.agent_positions: list[tuple[float, float]] = []
        self.cash = float(self.initial_capital)
        self.last_action = 0.0
        self.total_turnover = 0.0
        self.total_fees = 0.0
        self.profits = [0.0 for _ in range(len(self.data))]
        self.cumulative_return = [0.0 for _ in range(len(self.data))]
        self.prev_sr: Optional[float] = None
        self.ret_hist: list[float] = []
        self.episode_step = 0
        self.cooldown_remaining = 0
        self.last_buy_step: Optional[int] = None
        self.trade_attempt_count = 0
        self.executed_trade_count = 0
        self.blocked_trade_count = 0
        self.invalid_sell_count = 0

        self.init_price = self._current_price()
        self.equity_start = float(self.initial_capital)
        self.equity_end = float(self.initial_capital)
        self.prev_equity = float(self.initial_capital)
        self.realized_pnl = 0.0
        self.agent_open_position_value = 0.0

    def get_state(self) -> Optional[torch.Tensor]:
        if self.done:
            return None
        window = self.data.iloc[self.t - (self.window_size - 1) : self.t + 1]["Close"]
        return torch.tensor([float(value) for value in window], device=self.device, dtype=torch.float32)

    def get_account_features(self) -> dict[str, float]:
        price = self._current_price()
        equity = max(self._equity(price), self.sr_eps)
        holding_age = 0.0
        if self.last_buy_step is not None:
            holding_age = float(max(0, self.episode_step - self.last_buy_step))
        effective_max_exposure_ratio = self._effective_max_exposure_ratio()
        max_buy_quantity = self._max_buy_quantity(price)
        buy_headroom_ratio = float(np.clip((max_buy_quantity * price) / equity, 0.0, 1.0))
        min_hold_remaining = 0.0
        if self.min_hold_steps > 0 and self.last_buy_step is not None and self._position_units() > self.sr_eps:
            hold_duration = self.episode_step - self.last_buy_step
            min_hold_remaining = float(max(0, self.min_hold_steps - hold_duration))
        return {
            "cash_ratio": float(self.cash / equity),
            "position_ratio": float(self._position_value(price) / equity),
            "equity_return": float((equity / self.equity_start) - 1.0),
            "last_action": float(self.last_action),
            "turnover_ratio": float(self.total_turnover / max(self.equity_start, self.sr_eps)),
            "effective_max_exposure_ratio": float(
                effective_max_exposure_ratio if effective_max_exposure_ratio is not None else 0.0
            ),
            "buy_headroom_ratio": buy_headroom_ratio,
            "cooldown_remaining": float(self.cooldown_remaining),
            "min_hold_remaining": min_hold_remaining,
            "holding_age": holding_age,
        }

    def _reward_from_step_return(self, step_return: float, realized_profit_step: float) -> float:
        if self.reward_f == "profit":
            if realized_profit_step > 0:
                return 5.0
            if realized_profit_step < 0:
                return -5.0
            return 0.0
        if self.reward_f == "sr":
            self.ret_hist.append(float(step_return))
            if len(self.ret_hist) >= self.sr_window:
                window = np.array(self.ret_hist[-self.sr_window :], dtype=float)
                mu = float(np.mean(window))
                sigma = float(np.std(window, ddof=1))
                return float(self.sr_scale * mu / (sigma + self.sr_eps))
            return 0.0
        self.ret_hist.append(float(step_return))
        if len(self.ret_hist) < self.sr_window:
            reward = 0.0
        else:
            window = np.array(self.ret_hist[-self.sr_window :], dtype=float)
            mu = float(np.mean(window))
            sigma = float(np.std(window, ddof=1))
            sr_t = float(self.sr_scale * mu / (sigma + self.sr_eps))
            reward = 0.0 if self.prev_sr is None else sr_t - self.prev_sr
            self.prev_sr = sr_t
        if self.sr_clip is not None and self.sr_clip > 0:
            reward = float(np.clip(reward, -self.sr_clip, self.sr_clip))
        return float(reward)

    def step(self, act: int | torch.Tensor):
        if isinstance(act, torch.Tensor):
            act = int(act.item())
        if self.done:
            return (
                torch.tensor([0.0], device=self.device, dtype=torch.float32),
                True,
                torch.tensor([self._current_price()], device=self.device, dtype=torch.float32),
            )

        current_price = self._current_price()
        pre_equity = self._equity(current_price)
        if pre_equity <= self.sr_eps:
            self.done = True
            return (
                torch.tensor([-1.0], device=self.device, dtype=torch.float32),
                True,
                torch.tensor([current_price], device=self.device, dtype=torch.float32),
            )

        sell_nothing = False
        blocked_trade = False
        trade_executed = False
        realized_profit_step = 0.0
        blocked_by_cooldown = self.cooldown_remaining > 0
        if blocked_by_cooldown:
            self.cooldown_remaining -= 1

        buy_fraction: Optional[float] = None
        sell_fraction: Optional[float] = None
        is_sell_action = False
        if self.use_fractional_actions:
            buy_actions = len(self.buy_fractions)
            sell_actions = len(self.sell_fractions)
            if 1 <= act <= buy_actions:
                buy_fraction = self.buy_fractions[act - 1]
            elif (buy_actions + 1) <= act <= (buy_actions + sell_actions):
                sell_fraction = self.sell_fractions[act - buy_actions - 1]
        else:
            if act == 1:
                buy_fraction = 1.0
            elif act in {2, 3}:
                is_sell_action = True

        attempted_trade = buy_fraction is not None or sell_fraction is not None or is_sell_action
        if attempted_trade:
            self.trade_attempt_count += 1

        if buy_fraction is not None:
            if blocked_by_cooldown:
                blocked_trade = True
            else:
                position_before = self._position_units()
                max_buy_qty = self._max_buy_quantity(current_price)
                if max_buy_qty > self.sr_eps:
                    if self.buy_fractions:
                        buy_qty = max_buy_qty * buy_fraction
                    else:
                        buy_qty = 1.0 if max_buy_qty >= (1.0 - 1e-12) else 0.0
                    self._execute_buy(current_price, min(buy_qty, max_buy_qty))
                position_after = self._position_units()
                if position_after > position_before + self.sr_eps:
                    trade_executed = True
                    self.last_buy_step = self.episode_step

        elif sell_fraction is not None:
            if blocked_by_cooldown:
                blocked_trade = True
            elif self._position_units() <= self.sr_eps:
                sell_nothing = True
            else:
                if self.min_hold_steps > 0 and self.last_buy_step is not None:
                    hold_duration = self.episode_step - self.last_buy_step
                    if hold_duration < self.min_hold_steps:
                        blocked_trade = True
                if not blocked_trade:
                    position_before = self._position_units()
                    sell_qty = position_before * float(sell_fraction)
                    realized_profit_step = self._execute_sell(current_price, sell_qty)
                    self.profits[self.t] = realized_profit_step
                    position_after = self._position_units()
                    if position_after < position_before - self.sr_eps:
                        trade_executed = True
                    if position_after <= self.sr_eps:
                        self.last_buy_step = None

        elif is_sell_action:
            if blocked_by_cooldown:
                blocked_trade = True
            elif self._position_units() <= self.sr_eps:
                sell_nothing = True
            else:
                if self.min_hold_steps > 0 and self.last_buy_step is not None:
                    hold_duration = self.episode_step - self.last_buy_step
                    if hold_duration < self.min_hold_steps:
                        blocked_trade = True
                if not blocked_trade:
                    sell_all = False
                    if act == 2:
                        sell_all = self.sell_mode in {"all", "all_cap"}
                    elif act == 3:
                        if self.sell_mode == "one_plus":
                            sell_all = True
                        else:
                            sell_all = self.sell_mode in {"all", "all_cap"}

                    position_before = self._position_units()
                    sell_qty = position_before if sell_all else min(1.0, position_before)
                    realized_profit_step = self._execute_sell(current_price, sell_qty)
                    self.profits[self.t] = realized_profit_step
                    position_after = self._position_units()
                    if position_after < position_before - self.sr_eps:
                        trade_executed = True
                    if position_after <= self.sr_eps:
                        self.last_buy_step = None

        if trade_executed and self.trade_cooldown_steps > 0:
            self.cooldown_remaining = self.trade_cooldown_steps
        if self._position_units() <= self.sr_eps:
            self.last_buy_step = None
        if trade_executed:
            self.executed_trade_count += 1
        if blocked_trade:
            self.blocked_trade_count += 1
        if sell_nothing:
            self.invalid_sell_count += 1

        self.last_action = float(act)
        self.agent_open_position_value = float(
            sum((current_price - entry) * quantity for entry, quantity in self.agent_positions)
        )

        post_equity = self._equity(current_price)
        self.realized_pnl = post_equity - self.equity_start
        self.equity_end = post_equity
        step_return = (post_equity - self.prev_equity) / max(self.prev_equity, self.sr_eps)
        reward = self._reward_from_step_return(step_return, realized_profit_step)
        if sell_nothing and self.invalid_sell_penalty > 0:
            reward = min(float(reward), -self.invalid_sell_penalty)
        if blocked_trade and self.blocked_trade_penalty > 0:
            reward = float(reward) - self.blocked_trade_penalty
        self.prev_equity = post_equity
        self.cumulative_return[self.t] = (post_equity / max(self.equity_start, self.sr_eps)) - 1.0
        self.episode_step += 1

        if self.stop_on_bankruptcy and post_equity <= self.equity_start * self.min_equity_ratio:
            self.done = True
        if reward < -100:
            self.done = True

        self.t += 1
        if self._episode_end_index is not None and self.t >= self._episode_end_index:
            self.done = True

        return (
            torch.tensor([reward], device=self.device, dtype=torch.float32),
            self.done,
            torch.tensor([current_price], device=self.device, dtype=torch.float32),
        )
