from __future__ import annotations

from torch import nn


def _build_mlp_layers(input_dim: int, hidden_sizes: list[int], output_dim: int) -> nn.Sequential:
    layers = []
    current_dim = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(current_dim, size))
        layers.append(nn.LeakyReLU())
        current_dim = size
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


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
    def __init__(self, obs_dim: int, actions_n: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        self.net = _build_mlp_layers(obs_dim, hidden_sizes, actions_n)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class MLPDuelingDQN(nn.Module):
    def __init__(self, obs_dim: int, actions_n: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        self.feature_layer = _build_mlp_layers(obs_dim, hidden_sizes, hidden_sizes[-1])
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[-1], 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[-1], actions_n),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals


def build_q_network(
    model: str,
    input_dim: int,
    action_number: int,
    hidden_sizes: list[int] | None = None,
):
    hidden_sizes = hidden_sizes or [256, 256]
    if model in {"ddqn", "conv_dueling"}:
        return ConvDuelingDQN(input_dim, action_number)
    if model in {"dqn", "conv"}:
        return ConvDQN(input_dim, action_number)
    if model == "mlp":
        return MLPDQN(input_dim, action_number, hidden_sizes)
    if model == "mlp_dueling":
        return MLPDuelingDQN(input_dim, action_number, hidden_sizes)
    raise ValueError(f"Unsupported model type: {model}")
