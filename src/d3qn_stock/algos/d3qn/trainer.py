from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from d3qn_stock.algos.d3qn.agent import D3QNAgent
from d3qn_stock.envs.make_env import filter_date_range, load_price_data, make_env, sample_train_test_split
from d3qn_stock.utils.checkpoint import load_checkpoint, save_checkpoint
from d3qn_stock.utils.eval_metrics import (
    compute_active_streak_lengths,
    compute_annualized_sharpe_ratio,
    compute_max_drawdown_from_equity,
    compute_sharpe_ratio,
    safe_rate,
)
from d3qn_stock.utils.logging import LogPaths, MetricsLogger, setup_run_logger
from d3qn_stock.utils.path import RunPaths
from d3qn_stock.utils.seed import seed_everything


@dataclass
class DataConfig:
    path: str = "data/sample_stock.csv"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    date_column: str = "Date"
    close_column: str = "Close"
    open_column: Optional[str] = "Open"
    high_column: Optional[str] = "High"
    low_column: Optional[str] = "Low"
    volume_column: Optional[str] = "Volume"


@dataclass
class EnvConfig:
    reward: str = "profit"
    window_size: int = 10
    observation_mode: str = "raw"
    include_account_features: bool = False
    trading_period: Optional[int] = 24
    train_split: float = 0.8
    initial_capital: float = 10_000.0
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 1.0
    buy_fractions: list[float] = field(default_factory=lambda: [0.5, 1.0])
    sell_fractions: list[float] = field(default_factory=lambda: [0.5, 1.0])
    max_positions: Optional[int] = None
    max_exposure_ratio: Optional[float] = 1.0
    sell_mode: str = "all"
    invalid_sell_penalty: float = 0.1
    blocked_trade_penalty: float = 0.0
    min_hold_steps: int = 0
    trade_cooldown_steps: int = 0
    dynamic_exposure_enabled: bool = False
    dynamic_exposure_vol_window: int = 20
    dynamic_exposure_min_scale: float = 0.5
    dynamic_exposure_strength: float = 1.0
    min_equity_ratio: float = 0.2
    stop_on_bankruptcy: bool = True
    sr_window: int = 20
    sr_clip: float = 1.0
    periods_per_year: float = 252.0


@dataclass
class AgentConfig:
    replay_mem_size: int = 10_000
    batch_size: int = 32
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_steps: int = 2_000
    learning_rate: float = 0.0005
    input_dim: int = 10
    hidden_dim: int = 120
    action_number: int = 5
    target_update: int = 5
    model: str = "ddqn"
    double: bool = True
    per_enabled: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_steps: int = 10_000
    per_eps: float = 1e-6
    n_step: int = 3


@dataclass
class TrainConfig:
    num_episodes: int = 4
    max_steps_per_episode: Optional[int] = None
    max_total_steps: Optional[int] = None
    log_interval: int = 1
    checkpoint_interval: int = 2
    eval_epsilon: float = 0.0
    resample_train_window_each_episode: bool = False
    eval_interval: int = 0
    eval_episodes: int = 10
    eval_seed: Optional[int] = 20240101


@dataclass
class EvalConfig:
    num_episodes: int = 5
    seed: int = 20240101
    fixed_windows: bool = True
    fixed_windows_seed: Optional[int] = 20240101
    epsilon: float = 0.0
    save_per_episode: bool = True


@dataclass
class RunConfig:
    seed: int = 42
    device: str = "auto"


@dataclass
class ModelConfig:
    type: str = "mlp_dueling"
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    use_noisy: bool = False
    noisy_sigma_init: float = 0.5
    noisy_head_only: bool = False


@dataclass
class Config:
    data: DataConfig
    env: EnvConfig
    agent: AgentConfig
    train: TrainConfig
    eval: EvalConfig
    run: RunConfig
    model: ModelConfig


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _compute_start_range(
    data_len: int,
    window_size: int,
    trading_period: Optional[int],
) -> Tuple[int, int]:
    min_start = window_size - 1
    max_start = data_len - 1 if trading_period is None else data_len - trading_period
    return min_start, max_start


def _compute_fixed_eval_indices(
    data_len: int,
    window_size: int,
    trading_period: Optional[int],
    num_episodes: int,
    seed: int,
) -> List[int]:
    min_start, max_start = _compute_start_range(data_len, window_size, trading_period)
    if max_start < min_start:
        raise ValueError(
            f"Invalid start_index range [{min_start}, {max_start}] for data length {data_len} "
            f"and trading_period {trading_period}."
        )
    rng = np.random.default_rng(seed)
    return rng.integers(min_start, max_start + 1, size=num_episodes).tolist()


def config_from_dict(data: Dict) -> Config:
    return Config(
        data=DataConfig(**data.get("data", {})),
        env=EnvConfig(**data.get("env", {})),
        agent=AgentConfig(**data.get("agent", {})),
        train=TrainConfig(**data.get("train", {})),
        eval=EvalConfig(**data.get("eval", {})),
        run=RunConfig(**data.get("run", {})),
        model=ModelConfig(**data.get("model", {})),
    )


def config_to_dict(config: Config) -> Dict:
    return asdict(config)


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    raw = raw or {}
    return config_from_dict(raw)


def save_config(config: Config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config_to_dict(config), sort_keys=False))


def build_agent(config: Config, device: str, input_dim: Optional[int] = None) -> D3QNAgent:
    resolved_input_dim = input_dim if input_dim is not None else config.agent.input_dim
    return D3QNAgent(
        replay_mem_size=config.agent.replay_mem_size,
        batch_size=config.agent.batch_size,
        gamma=config.agent.gamma,
        eps_start=config.agent.eps_start,
        eps_end=config.agent.eps_end,
        eps_steps=config.agent.eps_steps,
        learning_rate=config.agent.learning_rate,
        input_dim=resolved_input_dim,
        hidden_dim=config.agent.hidden_dim,
        hidden_sizes=config.model.hidden_sizes,
        action_number=config.agent.action_number,
        target_update=config.agent.target_update,
        model=config.model.type,
        double=config.agent.double,
        per_enabled=config.agent.per_enabled,
        per_alpha=config.agent.per_alpha,
        per_beta_start=config.agent.per_beta_start,
        per_beta_steps=config.agent.per_beta_steps,
        per_eps=config.agent.per_eps,
        n_step=config.agent.n_step,
        use_noisy=config.model.use_noisy,
        noisy_sigma_init=config.model.noisy_sigma_init,
        noisy_head_only=config.model.noisy_head_only,
        device=device,
    )


def _prepare_data(config: Config):
    df = load_price_data(
        Path(config.data.path),
        close_column=config.data.close_column,
        date_column=config.data.date_column,
        open_column=config.data.open_column,
        high_column=config.data.high_column,
        low_column=config.data.low_column,
        volume_column=config.data.volume_column,
    )
    df = filter_date_range(
        df,
        date_column="Date",
        start_date=config.data.start_date,
        end_date=config.data.end_date,
    )
    return df


def _resolve_obs_dim(env, config: Config) -> int:
    obs_dim = getattr(env, "obs_dim", config.env.window_size)
    config.agent.input_dim = obs_dim
    return obs_dim


def _unwrap_trading_env(env):
    current = env
    for _ in range(10):
        if hasattr(current, "agent_positions"):
            return current
        if hasattr(current, "env"):
            current = current.env
            continue
        break
    return None


def _extract_account_features(base_env) -> dict[str, float]:
    if base_env is None or not hasattr(base_env, "get_account_features"):
        return {}
    return {key: float(value) for key, value in getattr(base_env, "get_account_features")().items()}


def _summarize_episode_diagnostics(
    base_env,
    equity_curve: list[float],
    position_ratios: list[float],
    effective_max_exposure_ratios: list[float],
) -> dict[str, float]:
    holding_streaks = compute_active_streak_lengths(position_ratios)
    turnover_ratio = 0.0
    if base_env is not None and hasattr(base_env, "total_turnover") and hasattr(base_env, "equity_start"):
        turnover_ratio = float(
            getattr(base_env, "total_turnover") / max(float(getattr(base_env, "equity_start")), 1e-8)
        )
    exposure_cap_hit_rate = 0.0
    if position_ratios and effective_max_exposure_ratios:
        cap_hits = [
            float(position >= max(cap - 1e-6, 0.0))
            for position, cap in zip(position_ratios, effective_max_exposure_ratios)
            if cap > 0.0
        ]
        exposure_cap_hit_rate = float(np.mean(cap_hits)) if cap_hits else 0.0
    trade_attempt_count = int(getattr(base_env, "trade_attempt_count", 0)) if base_env is not None else 0
    blocked_trade_count = int(getattr(base_env, "blocked_trade_count", 0)) if base_env is not None else 0
    invalid_sell_count = int(getattr(base_env, "invalid_sell_count", 0)) if base_env is not None else 0
    executed_trade_count = int(getattr(base_env, "executed_trade_count", 0)) if base_env is not None else 0
    return {
        "max_drawdown": compute_max_drawdown_from_equity(equity_curve),
        "avg_holding_time_steps": float(np.mean(holding_streaks)) if holding_streaks else 0.0,
        "turnover_ratio": turnover_ratio,
        "blocked_trade_rate": safe_rate(blocked_trade_count, trade_attempt_count),
        "invalid_sell_rate": safe_rate(invalid_sell_count, trade_attempt_count),
        "executed_trade_rate": safe_rate(executed_trade_count, trade_attempt_count),
        "mean_position_ratio": float(np.mean(position_ratios)) if position_ratios else 0.0,
        "max_position_ratio": float(np.max(position_ratios)) if position_ratios else 0.0,
        "mean_effective_max_exposure_ratio": (
            float(np.mean(effective_max_exposure_ratios)) if effective_max_exposure_ratios else 0.0
        ),
        "exposure_cap_hit_rate": exposure_cap_hit_rate,
    }


def _build_eval_env(config: Config, df, device: str):
    return make_env(
        df,
        config.env.reward,
        config.env.window_size,
        device,
        observation_mode=config.env.observation_mode,
        include_account_features=config.env.include_account_features,
        trading_period=config.env.trading_period,
        max_positions=config.env.max_positions,
        max_exposure_ratio=config.env.max_exposure_ratio,
        sell_mode=config.env.sell_mode,
        buy_fractions=config.env.buy_fractions,
        sell_fractions=config.env.sell_fractions,
        action_number=config.agent.action_number,
        initial_capital=config.env.initial_capital,
        transaction_cost_bps=config.env.transaction_cost_bps,
        slippage_bps=config.env.slippage_bps,
        invalid_sell_penalty=config.env.invalid_sell_penalty,
        blocked_trade_penalty=config.env.blocked_trade_penalty,
        min_hold_steps=config.env.min_hold_steps,
        trade_cooldown_steps=config.env.trade_cooldown_steps,
        dynamic_exposure_enabled=config.env.dynamic_exposure_enabled,
        dynamic_exposure_vol_window=config.env.dynamic_exposure_vol_window,
        dynamic_exposure_min_scale=config.env.dynamic_exposure_min_scale,
        dynamic_exposure_strength=config.env.dynamic_exposure_strength,
        min_equity_ratio=config.env.min_equity_ratio,
        stop_on_bankruptcy=config.env.stop_on_bankruptcy,
        sr_window=config.env.sr_window,
        sr_clip=config.env.sr_clip,
        periods_per_year=config.env.periods_per_year,
    )


def _evaluate_with_agent(
    config: Config,
    agent: D3QNAgent,
    episodes: int,
    epsilon: float,
    device: str,
    eval_seed: Optional[int] = None,
    eval_indices: Optional[List[int]] = None,
    eval_df=None,
) -> Tuple[float, Dict[str, float], list]:
    seed = eval_seed if eval_seed is not None else config.run.seed
    seed_everything(seed)

    df = _prepare_data(config) if eval_df is None else eval_df
    min_start, max_start = _compute_start_range(len(df), config.env.window_size, config.env.trading_period)
    if max_start < min_start:
        raise ValueError(
            "Invalid start_index range "
            f"[{min_start}, {max_start}] for data length {len(df)} "
            f"and trading_period {config.env.trading_period}."
        )

    if eval_indices is not None:
        if len(eval_indices) < episodes:
            raise ValueError("eval_indices length must be >= episodes.")
        start_indices = [int(eval_indices[idx]) for idx in range(episodes)]
    else:
        rng = np.random.default_rng(seed)
        start_indices = rng.integers(min_start, max_start + 1, size=episodes).tolist()

    env = _build_eval_env(config, df, device)
    _resolve_obs_dim(env, config)

    returns: List[float] = []
    return_rates: List[float] = []
    cumulative_returns: List[list] = []
    max_drawdowns: List[float] = []
    avg_holding_times: List[float] = []
    turnover_ratios: List[float] = []
    blocked_trade_rates: List[float] = []
    invalid_sell_rates: List[float] = []
    executed_trade_rates: List[float] = []
    mean_position_ratios: List[float] = []
    max_position_ratios: List[float] = []
    mean_effective_max_exposure_ratios: List[float] = []
    exposure_cap_hit_rates: List[float] = []

    policy_was_training = agent.policy_net.training
    target_was_training = agent.target_net.training
    agent.policy_net.eval()
    agent.target_net.eval()
    try:
        with torch.no_grad():
            for start_index in start_indices:
                env.reset(start_index=start_index)
                agent.reset_episode()
                state = env.get_state()
                base_env = _unwrap_trading_env(env)
                episode_equity_curve: list[float] = []
                episode_position_ratios: list[float] = []
                episode_effective_max_exposure_ratios: list[float] = []
                if base_env is not None and hasattr(base_env, "equity_start"):
                    episode_equity_curve.append(float(base_env.equity_start))

                episode_return = 0.0
                while state is not None:
                    action = agent.select_action(state, training=False, epsilon_override=epsilon)
                    reward, done, _ = env.step(action)
                    episode_return += reward.item()
                    features = _extract_account_features(base_env)
                    if features:
                        episode_position_ratios.append(float(features.get("position_ratio", 0.0)))
                        episode_effective_max_exposure_ratios.append(
                            float(features.get("effective_max_exposure_ratio", 0.0))
                        )
                    if base_env is not None and hasattr(base_env, "equity_end"):
                        episode_equity_curve.append(float(base_env.equity_end))
                    state = env.get_state()
                    if done:
                        break

                returns.append(episode_return)
                base_env = _unwrap_trading_env(env)
                if base_env is not None and hasattr(base_env, "equity_start") and hasattr(base_env, "equity_end"):
                    equity_start = float(base_env.equity_start)
                    equity_end = float(base_env.equity_end)
                    episode_return_rate = (equity_end / (equity_start + 1e-8)) - 1.0
                else:
                    episode_return_rate = 0.0
                if episode_equity_curve:
                    start_equity = max(float(episode_equity_curve[0]), 1e-8)
                    episode_cumulative_return_curve = [
                        float((equity / start_equity) - 1.0) for equity in episode_equity_curve
                    ]
                else:
                    episode_cumulative_return_curve = [0.0]
                cumulative_returns.append(episode_cumulative_return_curve)
                return_rates.append(episode_return_rate)
                diagnostics = _summarize_episode_diagnostics(
                    base_env,
                    episode_equity_curve,
                    episode_position_ratios,
                    episode_effective_max_exposure_ratios,
                )
                max_drawdowns.append(diagnostics["max_drawdown"])
                avg_holding_times.append(diagnostics["avg_holding_time_steps"])
                turnover_ratios.append(diagnostics["turnover_ratio"])
                blocked_trade_rates.append(diagnostics["blocked_trade_rate"])
                invalid_sell_rates.append(diagnostics["invalid_sell_rate"])
                executed_trade_rates.append(diagnostics["executed_trade_rate"])
                mean_position_ratios.append(diagnostics["mean_position_ratio"])
                max_position_ratios.append(diagnostics["max_position_ratio"])
                mean_effective_max_exposure_ratios.append(diagnostics["mean_effective_max_exposure_ratio"])
                exposure_cap_hit_rates.append(diagnostics["exposure_cap_hit_rate"])
    finally:
        if policy_was_training:
            agent.policy_net.train()
        if target_was_training:
            agent.target_net.train()

    mean_return = float(np.mean(returns)) if returns else 0.0
    median_return = float(np.median(returns)) if returns else 0.0
    std_return = float(np.std(returns)) if returns else 0.0
    mean_return_rate = float(np.mean(return_rates)) if return_rates else 0.0
    std_return_rate = float(np.std(return_rates)) if return_rates else 0.0
    metrics = {
        "mean_reward_return": mean_return,
        "median_reward_return": median_return,
        "std_reward_return": std_return,
        "mean_return_rate": mean_return_rate,
        "sharpe_ratio": compute_sharpe_ratio(mean_return_rate, std_return_rate),
        "annualized_sharpe_ratio": compute_annualized_sharpe_ratio(
            mean_return_rate,
            std_return_rate,
            config.env.periods_per_year,
        ),
        "win_rate": float(np.mean(np.array(returns) > 0.0)) if returns else 0.0,
        "max_drawdown": float(np.mean(max_drawdowns)) if max_drawdowns else 0.0,
        "avg_holding_time_steps": float(np.mean(avg_holding_times)) if avg_holding_times else 0.0,
        "turnover_ratio": float(np.mean(turnover_ratios)) if turnover_ratios else 0.0,
        "blocked_trade_rate": float(np.mean(blocked_trade_rates)) if blocked_trade_rates else 0.0,
        "invalid_sell_rate": float(np.mean(invalid_sell_rates)) if invalid_sell_rates else 0.0,
        "executed_trade_rate": float(np.mean(executed_trade_rates)) if executed_trade_rates else 0.0,
        "mean_position_ratio": float(np.mean(mean_position_ratios)) if mean_position_ratios else 0.0,
        "max_position_ratio": float(np.mean(max_position_ratios)) if max_position_ratios else 0.0,
        "mean_effective_max_exposure_ratio": (
            float(np.mean(mean_effective_max_exposure_ratios)) if mean_effective_max_exposure_ratios else 0.0
        ),
        "exposure_cap_hit_rate": float(np.mean(exposure_cap_hit_rates)) if exposure_cap_hit_rates else 0.0,
        "episodes": float(episodes),
        "epsilon": float(epsilon),
        "periods_per_year": float(config.env.periods_per_year),
    }
    return mean_return, metrics, cumulative_returns


def train(config: Config, run_paths: RunPaths) -> RunPaths:
    seed_everything(config.run.seed)
    device = _resolve_device(config.run.device)
    run_logger = setup_run_logger("train", run_paths.run_dir)
    run_logger.info("Logging dir: %s", run_paths.run_dir)

    df = _prepare_data(config)

    def _build_train_env():
        if config.env.trading_period is None:
            train_df = df
        else:
            train_df, _ = sample_train_test_split(
                df,
                trading_period=config.env.trading_period,
                train_split=config.env.train_split,
            )
        return make_env(
            train_df,
            config.env.reward,
            config.env.window_size,
            device,
            observation_mode=config.env.observation_mode,
            include_account_features=config.env.include_account_features,
            max_positions=config.env.max_positions,
            max_exposure_ratio=config.env.max_exposure_ratio,
            sell_mode=config.env.sell_mode,
            buy_fractions=config.env.buy_fractions,
            sell_fractions=config.env.sell_fractions,
            action_number=config.agent.action_number,
            initial_capital=config.env.initial_capital,
            transaction_cost_bps=config.env.transaction_cost_bps,
            slippage_bps=config.env.slippage_bps,
            invalid_sell_penalty=config.env.invalid_sell_penalty,
            blocked_trade_penalty=config.env.blocked_trade_penalty,
            min_hold_steps=config.env.min_hold_steps,
            trade_cooldown_steps=config.env.trade_cooldown_steps,
            dynamic_exposure_enabled=config.env.dynamic_exposure_enabled,
            dynamic_exposure_vol_window=config.env.dynamic_exposure_vol_window,
            dynamic_exposure_min_scale=config.env.dynamic_exposure_min_scale,
            dynamic_exposure_strength=config.env.dynamic_exposure_strength,
            min_equity_ratio=config.env.min_equity_ratio,
            stop_on_bankruptcy=config.env.stop_on_bankruptcy,
            sr_window=config.env.sr_window,
            sr_clip=config.env.sr_clip,
            periods_per_year=config.env.periods_per_year,
        )

    env = _build_train_env()
    obs_dim = _resolve_obs_dim(env, config)
    agent = build_agent(config, device, input_dim=obs_dim)

    log_paths = LogPaths(
        run_dir=run_paths.run_dir,
        metrics_csv=run_paths.metrics_csv,
        tensorboard_dir=run_paths.tensorboard_dir,
    )
    metrics_logger = MetricsLogger(
        log_paths,
        fieldnames=["episode", "steps", "reward_return", "epsilon", "avg_loss", "avg_q"],
    )

    save_config(config, run_paths.config_resolved)

    total_steps = 0
    stop_training = False
    eval_history: List[Dict[str, float]] = []
    eval_seed = getattr(config.train, "eval_seed", None) or config.eval.seed
    eval_interval = getattr(config.train, "eval_interval", 0)
    eval_episodes = getattr(config.train, "eval_episodes", config.eval.num_episodes)

    fixed_eval_indices: Optional[List[int]] = None
    if eval_interval > 0 and eval_episodes > 0:
        fixed_eval_indices = _compute_fixed_eval_indices(
            len(df),
            config.env.window_size,
            config.env.trading_period,
            eval_episodes,
            eval_seed,
        )

    for episode in range(config.train.num_episodes):
        if episode > 0 and config.train.resample_train_window_each_episode:
            env = _build_train_env()
            current_obs_dim = getattr(env, "obs_dim", config.env.window_size)
            if current_obs_dim != obs_dim:
                raise ValueError("Observation dimension changed after train-window resampling.")
        env.reset(start_index=env.window_size - 1)
        agent.reset_episode()
        state = env.get_state()
        reward_return = 0.0
        losses = []
        q_values = []
        steps = 0
        episode_done = False

        while state is not None:
            action = agent.select_action(state, training=True)
            reward, done, _ = env.step(action)
            reward_return += reward.item()
            next_state = env.get_state()
            agent.store_transition(state, action, next_state, reward, done=done)

            result = agent.optimize_double_dqn() if agent.double else agent.optimize()
            if result:
                loss_value, q_value = result
                losses.append(loss_value)
                q_values.append(q_value)

            state = next_state
            steps += 1
            total_steps += 1
            if config.train.max_total_steps and total_steps >= config.train.max_total_steps:
                stop_training = True
                break
            if config.train.max_steps_per_episode and steps >= config.train.max_steps_per_episode:
                break
            if done:
                episode_done = True
                break

        if not episode_done:
            agent.finalize_episode(force_terminal=False)

        if episode % agent.target_update == 0:
            agent.update_target()

        avg_loss = float(np.mean(losses)) if losses else 0.0
        avg_q = float(np.mean(q_values)) if q_values else 0.0
        metrics_logger.log(
            {
                "episode": episode,
                "steps": steps,
                "reward_return": reward_return,
                "epsilon": agent.last_epsilon,
                "avg_loss": avg_loss,
                "avg_q": avg_q,
            },
            step=episode,
        )

        if (episode + 1) % config.train.log_interval == 0:
            run_logger.info(
                "Episode %s/%s | reward_return %.2f | epsilon %.3f | avg_loss %.4f | avg_q %.4f",
                episode + 1,
                config.train.num_episodes,
                reward_return,
                agent.last_epsilon,
                avg_loss,
                avg_q,
            )

        if (episode + 1) % config.train.checkpoint_interval == 0:
            ckpt_path = run_paths.checkpoints_dir / f"episode_{episode + 1}.pt"
            save_checkpoint(
                ckpt_path,
                policy_state=agent.policy_net.state_dict(),
                target_state=agent.target_net.state_dict(),
                optimizer_state=agent.optimizer.state_dict(),
                config=config_to_dict(config),
                episode=episode + 1,
                step=total_steps,
            )

        if eval_interval > 0 and eval_episodes > 0 and (episode + 1) % eval_interval == 0:
            _, eval_metrics, _ = _evaluate_with_agent(
                config,
                agent,
                episodes=eval_episodes,
                epsilon=0.0,
                device=device,
                eval_seed=eval_seed,
                eval_indices=fixed_eval_indices,
                eval_df=df,
            )
            eval_history.append(
                {
                    "eval_at_episode": float(episode + 1),
                    "mean_reward_return": eval_metrics["mean_reward_return"],
                    "median_reward_return": eval_metrics["median_reward_return"],
                    "std_reward_return": eval_metrics["std_reward_return"],
                    "mean_return_rate": eval_metrics["mean_return_rate"] * 100.0,
                    "sharpe_ratio": eval_metrics["sharpe_ratio"],
                    "annualized_sharpe_ratio": eval_metrics["annualized_sharpe_ratio"],
                    "win_rate": eval_metrics["win_rate"],
                }
            )

        if stop_training:
            break

    final_ckpt = run_paths.checkpoints_dir / "checkpoint_latest.pt"
    save_checkpoint(
        final_ckpt,
        policy_state=agent.policy_net.state_dict(),
        target_state=agent.target_net.state_dict(),
        optimizer_state=agent.optimizer.state_dict(),
        config=config_to_dict(config),
        episode=config.train.num_episodes,
        step=total_steps,
    )

    if eval_history:
        eval_csv = run_paths.run_dir / "eval_history.csv"
        with eval_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "eval_at_episode",
                    "mean_reward_return",
                    "median_reward_return",
                    "std_reward_return",
                    "mean_return_rate",
                    "sharpe_ratio",
                    "annualized_sharpe_ratio",
                    "win_rate",
                ],
            )
            writer.writeheader()
            writer.writerows(eval_history)
        run_logger.info("Eval history written to %s", eval_csv)

    metrics_logger.close()
    run_logger.info("Run artifacts saved to %s", run_paths.run_dir)
    return run_paths


def evaluate(
    config: Config,
    checkpoint_path: Path,
    episodes: int,
    epsilon: float,
    device: str,
    eval_seed: Optional[int] = None,
    eval_indices: Optional[List[int]] = None,
) -> Tuple[float, Dict[str, float], list]:
    device = _resolve_device(device)

    df = _prepare_data(config)
    dim_env = _build_eval_env(config, df, device)
    obs_dim = _resolve_obs_dim(dim_env, config)
    agent = build_agent(config, device, input_dim=obs_dim)

    checkpoint = load_checkpoint(checkpoint_path, device=device)
    agent.policy_net.load_state_dict(checkpoint["policy_state"])
    agent.target_net.load_state_dict(checkpoint["target_state"])
    return _evaluate_with_agent(
        config,
        agent,
        episodes=episodes,
        epsilon=epsilon,
        device=device,
        eval_seed=eval_seed,
        eval_indices=eval_indices,
        eval_df=df,
    )
