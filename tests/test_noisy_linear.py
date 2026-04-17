from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from d3qn_stock.algos.d3qn.networks import MLPDuelingDQN, NoisyLinear


class TestNoisyLinear(unittest.TestCase):
    def test_noisy_linear_changes_output_after_noise_reset_in_train_mode(self) -> None:
        layer = NoisyLinear(4, 3)
        layer.train()
        x = torch.ones(1, 4)
        first = layer(x)
        layer.reset_noise()
        second = layer(x)
        self.assertFalse(torch.allclose(first, second))

    def test_noisy_linear_is_deterministic_in_eval_mode(self) -> None:
        layer = NoisyLinear(4, 3)
        layer.eval()
        x = torch.ones(1, 4)
        first = layer(x)
        layer.reset_noise()
        second = layer(x)
        self.assertTrue(torch.allclose(first, second))

    def test_mlp_dueling_can_reset_noisy_layers(self) -> None:
        net = MLPDuelingDQN(obs_dim=6, actions_n=5, hidden_sizes=[8, 8], use_noisy=True)
        net.train()
        x = torch.randn(2, 1, 6)
        first = net(x)
        net.reset_noise()
        second = net(x)
        self.assertEqual(first.shape, second.shape)
        self.assertFalse(torch.allclose(first, second))

    def test_noisy_sigma_init_sets_expected_weight_scale(self) -> None:
        layer = NoisyLinear(4, 3, sigma_init=0.2)
        expected = 0.2 / (4 ** 0.5)
        self.assertTrue(torch.allclose(layer.weight_sigma, torch.full_like(layer.weight_sigma, expected)))

    def test_mlp_dueling_head_only_leaves_feature_extractor_deterministic(self) -> None:
        net = MLPDuelingDQN(
            obs_dim=6,
            actions_n=5,
            hidden_sizes=[8, 8],
            use_noisy=True,
            noisy_sigma_init=0.2,
            noisy_head_only=True,
        )
        feature_noisy = any(isinstance(module, NoisyLinear) for module in net.feature_layer.modules())
        value_noisy = any(isinstance(module, NoisyLinear) for module in net.value_stream.modules())
        advantage_noisy = any(isinstance(module, NoisyLinear) for module in net.advantage_stream.modules())
        self.assertFalse(feature_noisy)
        self.assertTrue(value_noisy)
        self.assertTrue(advantage_noisy)

    def test_mlp_head_only_uses_only_output_noisy_layer(self) -> None:
        from d3qn_stock.algos.d3qn.networks import MLPDQN

        net = MLPDQN(
            obs_dim=6,
            actions_n=5,
            hidden_sizes=[8, 8],
            use_noisy=True,
            noisy_sigma_init=0.1,
            noisy_head_only=True,
        )
        feature_noisy = any(isinstance(module, NoisyLinear) for module in net.feature_layer.modules())
        output_noisy = isinstance(net.output_layer, NoisyLinear)
        self.assertFalse(feature_noisy)
        self.assertTrue(output_noisy)


if __name__ == "__main__":
    unittest.main()
