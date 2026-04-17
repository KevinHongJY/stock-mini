from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class NoisyLinear(nn.Module):
    """Factorized Gaussian noisy linear layer from NoisyNet."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int, device: torch.device) -> torch.Tensor:
        noise = torch.randn(size, device=device)
        return noise.sign() * noise.abs().sqrt()

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features, self.weight_mu.device)
        eps_out = self._scale_noise(self.out_features, self.weight_mu.device)
        self.weight_epsilon.copy_(torch.outer(eps_out, eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


def _build_linear_layer(
    input_dim: int,
    output_dim: int,
    use_noisy: bool,
    noisy_sigma_init: float = 0.5,
) -> nn.Module:
    if use_noisy:
        return NoisyLinear(input_dim, output_dim, sigma_init=noisy_sigma_init)
    return nn.Linear(input_dim, output_dim)


def _build_mlp_layers(
    input_dim: int,
    hidden_sizes: list[int],
    output_dim: int,
    use_noisy: bool = False,
    noisy_sigma_init: float = 0.5,
) -> nn.Sequential:
    layers = []
    current_dim = input_dim
    for size in hidden_sizes:
        layers.append(
            _build_linear_layer(
                current_dim,
                size,
                use_noisy=use_noisy,
                noisy_sigma_init=noisy_sigma_init,
            )
        )
        layers.append(nn.LeakyReLU())
        current_dim = size
    layers.append(
        _build_linear_layer(
            current_dim,
            output_dim,
            use_noisy=use_noisy,
            noisy_sigma_init=noisy_sigma_init,
        )
    )
    return nn.Sequential(*layers)


def _reset_noisy_layers(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, NoisyLinear):
            child.reset_noise()


class DQN(nn.Module):
    def __init__(self, obs_len: int, hidden_size: int, actions_n: int) -> None:
        super().__init__()
        self.fc_val = nn.Sequential(
            nn.Linear(obs_len, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, actions_n),
        )

    def forward(self, x):
        return self.fc_val(x)


class DuelingDQN(nn.Module):
    def __init__(self, obs_len: int, hidden_size: int, actions_n: int) -> None:
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(obs_len, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, actions_n),
        )

    def forward(self, state):
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals


class ConvDQN(nn.Module):
    def __init__(self, seq_len_in: int, actions_n: int, kernel_size: int = 8) -> None:
        super().__init__()
        n_filters = 64
        max_pool_kernel = 2
        self.conv1 = nn.Conv1d(1, n_filters, kernel_size)
        self.max_pool = nn.MaxPool1d(max_pool_kernel, stride=1)
        self.activation = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size // 2)
        self.hidden_dim = n_filters * (
            ((((seq_len_in - kernel_size + 1) - max_pool_kernel + 1) - kernel_size // 2 + 1) - max_pool_kernel + 1)
        )
        self.out_layer = nn.Linear(self.hidden_dim, actions_n)

    def forward(self, x):
        c1_out = self.conv1(x)
        max_pool_1 = self.max_pool(self.activation(c1_out))
        c2_out = self.conv2(max_pool_1)
        max_pool_2 = self.max_pool(self.activation(c2_out))
        max_pool_2 = max_pool_2.view(-1, self.hidden_dim)
        return self.activation(self.out_layer(max_pool_2))


class ConvDuelingDQN(nn.Module):
    def __init__(self, seq_len_in: int, actions_n: int, kernel_size: int = 8) -> None:
        super().__init__()
        n_filters = 64
        max_pool_kernel = 2
        self.conv1 = nn.Conv1d(1, n_filters, kernel_size)
        self.max_pool = nn.MaxPool1d(max_pool_kernel, stride=1)
        self.activation = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size // 2)
        self.hidden_dim = n_filters * (
            ((((seq_len_in - kernel_size + 1) - max_pool_kernel + 1) - kernel_size // 2 + 1) - max_pool_kernel + 1)
        )
        paper_hidden_dim = 120
        self.split_layer = nn.Linear(self.hidden_dim, paper_hidden_dim)
        self.value_stream = nn.Sequential(
            nn.Linear(paper_hidden_dim, paper_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(paper_hidden_dim, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(paper_hidden_dim, paper_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(paper_hidden_dim, actions_n),
        )

    def forward(self, x):
        c1_out = self.conv1(x)
        max_pool_1 = self.max_pool(self.activation(c1_out))
        c2_out = self.conv2(max_pool_1)
        max_pool_2 = self.max_pool(self.activation(c2_out))
        max_pool_2 = max_pool_2.view(-1, self.hidden_dim)
        split = self.split_layer(max_pool_2)
        values = self.value_stream(split)
        advantages = self.advantage_stream(split)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals


class MLPDQN(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        actions_n: int,
        hidden_sizes: list[int],
        use_noisy: bool = False,
        noisy_sigma_init: float = 0.5,
        noisy_head_only: bool = False,
    ) -> None:
        super().__init__()
        self.noisy_head_only = bool(use_noisy and noisy_head_only)
        if self.noisy_head_only:
            self.feature_layer = _build_mlp_layers(
                obs_dim,
                hidden_sizes,
                hidden_sizes[-1],
                use_noisy=False,
                noisy_sigma_init=noisy_sigma_init,
            )
            self.output_layer = _build_linear_layer(
                hidden_sizes[-1],
                actions_n,
                use_noisy=True,
                noisy_sigma_init=noisy_sigma_init,
            )
        else:
            self.net = _build_mlp_layers(
                obs_dim,
                hidden_sizes,
                actions_n,
                use_noisy=use_noisy,
                noisy_sigma_init=noisy_sigma_init,
            )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if self.noisy_head_only:
            return self.output_layer(self.feature_layer(x))
        return self.net(x)

    def reset_noise(self) -> None:
        _reset_noisy_layers(self)


class MLPDuelingDQN(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        actions_n: int,
        hidden_sizes: list[int],
        use_noisy: bool = False,
        noisy_sigma_init: float = 0.5,
        noisy_head_only: bool = False,
    ) -> None:
        super().__init__()
        feature_layer_noisy = bool(use_noisy and not noisy_head_only)
        self.feature_layer = _build_mlp_layers(
            obs_dim,
            hidden_sizes,
            hidden_sizes[-1],
            use_noisy=feature_layer_noisy,
            noisy_sigma_init=noisy_sigma_init,
        )
        self.value_stream = nn.Sequential(
            _build_linear_layer(
                hidden_sizes[-1],
                hidden_sizes[-1],
                use_noisy=use_noisy,
                noisy_sigma_init=noisy_sigma_init,
            ),
            nn.LeakyReLU(),
            _build_linear_layer(
                hidden_sizes[-1],
                1,
                use_noisy=use_noisy,
                noisy_sigma_init=noisy_sigma_init,
            ),
        )
        self.advantage_stream = nn.Sequential(
            _build_linear_layer(
                hidden_sizes[-1],
                hidden_sizes[-1],
                use_noisy=use_noisy,
                noisy_sigma_init=noisy_sigma_init,
            ),
            nn.LeakyReLU(),
            _build_linear_layer(
                hidden_sizes[-1],
                actions_n,
                use_noisy=use_noisy,
                noisy_sigma_init=noisy_sigma_init,
            ),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

    def reset_noise(self) -> None:
        _reset_noisy_layers(self)


def build_q_network(
    model: str,
    input_dim: int,
    action_number: int,
    hidden_sizes: list[int] | None = None,
    use_noisy: bool = False,
    noisy_sigma_init: float = 0.5,
    noisy_head_only: bool = False,
):
    hidden_sizes = hidden_sizes or [256, 256]
    if model in {"ddqn", "conv_dueling"}:
        return ConvDuelingDQN(input_dim, action_number)
    if model in {"dqn", "conv"}:
        return ConvDQN(input_dim, action_number)
    if model == "mlp":
        return MLPDQN(
            input_dim,
            action_number,
            hidden_sizes,
            use_noisy=use_noisy,
            noisy_sigma_init=noisy_sigma_init,
            noisy_head_only=noisy_head_only,
        )
    if model == "mlp_dueling":
        return MLPDuelingDQN(
            input_dim,
            action_number,
            hidden_sizes,
            use_noisy=use_noisy,
            noisy_sigma_init=noisy_sigma_init,
            noisy_head_only=noisy_head_only,
        )
    raise ValueError(f"Unsupported model type: {model}")
