from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from d3qn_stock.envs.trading_env_discrete_capital import DiscreteCapitalTradingEnvironment


def _build_df(prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=len(prices), freq="D"),
            "Close": prices,
        }
    )


class TestDiscreteCapitalTradingEnvironment(unittest.TestCase):
    def test_return_observation_uses_log_returns(self) -> None:
        env = DiscreteCapitalTradingEnvironment(
            _build_df([100.0, 110.0, 121.0, 133.1]),
            reward="profit",
            window_size=3,
            observation_mode="returns",
            device="cpu",
        )
        env.reset(start_index=2)

        state = env.get_state()

        expected = np.array([0.0, np.log(110.0 / 100.0), np.log(121.0 / 110.0)], dtype=np.float32)
        self.assertEqual(tuple(state.shape), (3,))
        self.assertTrue(np.allclose(state.cpu().numpy(), expected, atol=1e-6))

    def test_account_features_are_appended_to_observation(self) -> None:
        env = DiscreteCapitalTradingEnvironment(
            _build_df([100.0, 100.0, 100.0, 100.0]),
            reward="profit",
            window_size=2,
            include_account_features=True,
            device="cpu",
        )
        env.reset(start_index=1)

        state = env.get_state()

        self.assertEqual(env.obs_dim, 2 + len(env.ACCOUNT_FEATURE_KEYS))
        self.assertEqual(tuple(state.shape), (env.obs_dim,))
        state_np = state.cpu().numpy()
        self.assertTrue(np.allclose(state_np[:2], np.array([100.0, 100.0], dtype=np.float32)))
        self.assertAlmostEqual(float(state_np[2]), 1.0, places=6)
        self.assertAlmostEqual(float(state_np[3]), 0.0, places=6)
        self.assertAlmostEqual(float(state_np[4]), 0.0, places=6)

    def test_fractional_buy_and_sell_updates_cash_and_position(self) -> None:
        env = DiscreteCapitalTradingEnvironment(
            _build_df([100.0, 100.0, 100.0, 100.0, 100.0]),
            reward="profit",
            window_size=2,
            buy_fractions=[0.5, 1.0],
            sell_fractions=[0.5, 1.0],
            action_number=5,
            initial_capital=1000.0,
            transaction_cost_bps=0.0,
            slippage_bps=0.0,
            device="cpu",
        )
        env.reset(start_index=1)

        env.step(1)
        self.assertAlmostEqual(env.cash, 500.0, places=5)
        self.assertAlmostEqual(env._position_units(), 5.0, places=5)

        env.step(3)
        self.assertAlmostEqual(env.cash, 750.0, places=5)
        self.assertAlmostEqual(env._position_units(), 2.5, places=5)

    def test_transaction_cost_and_slippage_reduce_affordable_position(self) -> None:
        base_env = DiscreteCapitalTradingEnvironment(
            _build_df([100.0, 100.0, 100.0, 100.0]),
            reward="profit",
            window_size=2,
            buy_fractions=[1.0],
            sell_fractions=[1.0],
            action_number=3,
            initial_capital=1000.0,
            transaction_cost_bps=0.0,
            slippage_bps=0.0,
            device="cpu",
        )
        cost_env = DiscreteCapitalTradingEnvironment(
            _build_df([100.0, 100.0, 100.0, 100.0]),
            reward="profit",
            window_size=2,
            buy_fractions=[1.0],
            sell_fractions=[1.0],
            action_number=3,
            initial_capital=1000.0,
            transaction_cost_bps=100.0,
            slippage_bps=100.0,
            device="cpu",
        )
        base_env.reset(start_index=1)
        cost_env.reset(start_index=1)

        base_env.step(1)
        cost_env.step(1)

        self.assertGreater(base_env._position_units(), cost_env._position_units())
        self.assertGreater(cost_env.total_fees, 0.0)

    def test_invalid_sell_counts_and_penalizes(self) -> None:
        env = DiscreteCapitalTradingEnvironment(
            _build_df([100.0, 100.0, 100.0, 100.0]),
            reward="profit",
            window_size=2,
            buy_fractions=[1.0],
            sell_fractions=[1.0],
            action_number=3,
            initial_capital=1000.0,
            invalid_sell_penalty=0.2,
            transaction_cost_bps=0.0,
            slippage_bps=0.0,
            device="cpu",
        )
        env.reset(start_index=1)

        reward, done, _ = env.step(2)

        self.assertFalse(done)
        self.assertEqual(env.invalid_sell_count, 1)
        self.assertAlmostEqual(float(reward.item()), -0.2, places=6)

    def test_equity_increases_when_price_rises_after_buy(self) -> None:
        env = DiscreteCapitalTradingEnvironment(
            _build_df([100.0, 100.0, 110.0, 110.0, 110.0]),
            reward="profit",
            window_size=2,
            buy_fractions=[1.0],
            sell_fractions=[1.0],
            action_number=3,
            initial_capital=1000.0,
            transaction_cost_bps=0.0,
            slippage_bps=0.0,
            device="cpu",
        )
        env.reset(start_index=1)

        env.step(1)
        env.step(0)

        self.assertGreater(env.equity_end, env.equity_start)


if __name__ == "__main__":
    unittest.main()
