from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from d3qn_stock.algos.d3qn.agent import D3QNAgent


def _state(value: float) -> torch.Tensor:
    return torch.tensor([value, value + 1.0, value + 2.0, value + 3.0], dtype=torch.float32)


class TestD3QNNstepAndPER(unittest.TestCase):
    def _build_agent(self, per_enabled: bool = False, n_step: int = 3, batch_size: int = 2) -> D3QNAgent:
        return D3QNAgent(
            replay_mem_size=64,
            batch_size=batch_size,
            gamma=0.99,
            eps_start=0.0,
            eps_end=0.0,
            eps_steps=1,
            learning_rate=1e-3,
            input_dim=4,
            hidden_dim=16,
            hidden_sizes=[16, 16],
            action_number=3,
            target_update=10,
            model="mlp_dueling",
            double=True,
            per_enabled=per_enabled,
            per_alpha=0.6,
            per_beta_start=0.4,
            per_beta_steps=100,
            per_eps=1e-6,
            n_step=n_step,
            device="cpu",
        )

    def test_reset_episode_flushes_truncation_without_terminal_flag(self) -> None:
        agent = self._build_agent(per_enabled=False, n_step=3)

        s0 = _state(0.0)
        s1 = _state(1.0)
        s2 = _state(2.0)

        agent.store_transition(s0, 1, s1, 0.1, done=False)
        agent.store_transition(s1, 2, s2, 0.2, done=False)
        self.assertEqual(len(agent.memory), 0)

        agent.reset_episode()

        self.assertEqual(len(agent.memory), 2)
        for transition in agent.memory.memory:
            self.assertIsNotNone(transition.next_state)
            self.assertEqual(float(transition.done.item()), 0.0)

    def test_natural_terminal_still_marks_done(self) -> None:
        agent = self._build_agent(per_enabled=False, n_step=3)

        s0 = _state(10.0)
        s1 = _state(11.0)

        agent.store_transition(s0, 1, s1, 0.3, done=False)
        agent.store_transition(s1, 2, None, 0.4, done=True)

        self.assertEqual(len(agent.memory), 2)
        for transition in agent.memory.memory:
            self.assertIsNone(transition.next_state)
            self.assertEqual(float(transition.done.item()), 1.0)

    def test_per_optimize_updates_priorities(self) -> None:
        agent = self._build_agent(per_enabled=True, n_step=3, batch_size=2)

        for step in range(6):
            done = step == 5
            current_state = _state(float(step))
            next_state = None if done else _state(float(step + 1))
            action = step % 3
            reward = float(step) * 0.01
            agent.store_transition(current_state, action, next_state, reward, done=done)

        self.assertGreaterEqual(len(agent.memory), agent.batch_size)
        with patch.object(agent.memory, "update_priorities", wraps=agent.memory.update_priorities) as wrapped:
            result = agent.optimize_double_dqn()
            self.assertIsNotNone(result)
            self.assertTrue(wrapped.called)


if __name__ == "__main__":
    unittest.main()
