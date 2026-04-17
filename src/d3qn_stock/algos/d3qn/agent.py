from __future__ import annotations

import random
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

from d3qn_stock.algos.d3qn.losses import mse_q_loss, weighted_huber_q_loss
from d3qn_stock.algos.d3qn.networks import build_q_network
from d3qn_stock.algos.d3qn.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer, Transition
from d3qn_stock.algos.d3qn.schedules import EpsilonSchedule


class D3QNAgent:
    def __init__(
        self,
        replay_mem_size: int = 10000,
        batch_size: int = 40,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.02,
        eps_steps: int = 200,
        learning_rate: float = 0.0005,
        input_dim: int = 24,
        hidden_dim: int = 120,
        hidden_sizes: Optional[list[int]] = None,
        action_number: int = 3,
        target_update: int = 5,
        model: str = "ddqn",
        double: bool = True,
        per_enabled: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_steps: int = 100000,
        per_eps: float = 1e-6,
        n_step: int = 1,
        use_noisy: bool = False,
        noisy_sigma_init: float = 0.5,
        noisy_head_only: bool = False,
        device: str = "auto",
    ) -> None:
        if n_step < 1:
            raise ValueError("n_step must be >= 1.")

        self.replay_mem_size = replay_mem_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_sizes = hidden_sizes or [hidden_dim, hidden_dim]
        self.action_number = action_number
        self.target_update = target_update
        self.model = model
        self.double = double
        self.per_enabled = per_enabled
        self.per_alpha = per_alpha
        self.per_beta_start = per_beta_start
        self.per_beta_steps = max(int(per_beta_steps), 1)
        self.per_eps = per_eps
        self.n_step = int(n_step)
        self.n_step_discount = self.gamma ** self.n_step
        self.use_noisy = use_noisy
        self.noisy_sigma_init = noisy_sigma_init
        self.noisy_head_only = noisy_head_only
        self.training = True

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )

        self.policy_net = build_q_network(
            self.model,
            self.input_dim,
            self.action_number,
            hidden_sizes=self.hidden_sizes,
            use_noisy=self.use_noisy,
            noisy_sigma_init=self.noisy_sigma_init,
            noisy_head_only=self.noisy_head_only,
        ).to(self.device)
        self.target_net = build_q_network(
            self.model,
            self.input_dim,
            self.action_number,
            hidden_sizes=self.hidden_sizes,
            use_noisy=self.use_noisy,
            noisy_sigma_init=self.noisy_sigma_init,
            noisy_head_only=self.noisy_head_only,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        if self.per_enabled:
            self.memory = PrioritizedReplayBuffer(self.replay_mem_size, alpha=self.per_alpha)
        else:
            self.memory = ReplayBuffer(self.replay_mem_size)
        self.epsilon_schedule = EpsilonSchedule(self.eps_start, self.eps_end, self.eps_steps)
        self.n_step_buffer: Deque[Transition] = deque()
        self.steps_done = 0
        self.optimize_steps = 0
        self.last_epsilon = self.eps_start

    def _reset_policy_noise(self) -> None:
        if hasattr(self.policy_net, "reset_noise"):
            self.policy_net.reset_noise()

    def _as_tensor(self, value: Optional[torch.Tensor | np.ndarray]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(value, device=self.device, dtype=torch.float32)

    def _as_action_tensor(self, action: torch.Tensor | int) -> torch.Tensor:
        if isinstance(action, torch.Tensor):
            return action.to(device=self.device, dtype=torch.long).view(-1)
        return torch.tensor([int(action)], device=self.device, dtype=torch.long)

    def _as_reward_tensor(self, reward: torch.Tensor | float) -> torch.Tensor:
        if isinstance(reward, torch.Tensor):
            return reward.to(device=self.device, dtype=torch.float32).view(-1)
        return torch.tensor([float(reward)], device=self.device, dtype=torch.float32)

    def _as_done_tensor(self, done: torch.Tensor | bool) -> torch.Tensor:
        if isinstance(done, torch.Tensor):
            return done.to(device=self.device, dtype=torch.float32).view(-1)
        return torch.tensor([1.0 if bool(done) else 0.0], device=self.device, dtype=torch.float32)

    def reset_episode(self) -> None:
        if self.n_step > 1 and self.n_step_buffer:
            self.finalize_episode(force_terminal=False)
        if self.use_noisy:
            self._reset_policy_noise()

    def select_action(
        self,
        state: torch.Tensor,
        training: bool = True,
        epsilon_override: Optional[float] = None,
    ) -> torch.Tensor:
        state = self._as_tensor(state)
        if state is None:
            raise ValueError("State must not be None when selecting an action.")
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if state.dim() == 2:
            state = state.unsqueeze(1)
        if self.use_noisy:
            self.last_epsilon = 0.0
            self.steps_done += 1
            if training:
                self.policy_net.train()
                self._reset_policy_noise()
            else:
                self.policy_net.eval()
            with torch.no_grad():
                return torch.tensor([self.policy_net(state).argmax()], device=self.device, dtype=torch.long)
        sample = random.random()
        if epsilon_override is not None:
            eps_threshold = epsilon_override
        else:
            eps_threshold = self.epsilon_schedule.value(self.steps_done, training=training)
        self.last_epsilon = eps_threshold
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return torch.tensor([self.policy_net(state).argmax()], device=self.device, dtype=torch.long)
        return torch.tensor([random.randrange(self.action_number)], device=self.device, dtype=torch.long)

    def _build_n_step_transition(self, force_terminal: bool = False) -> Transition:
        if not self.n_step_buffer:
            raise ValueError("n_step_buffer is empty.")

        first = self.n_step_buffer[0]
        reward_n = torch.zeros_like(first.reward)
        next_state_n = first.next_state
        done_n = torch.zeros_like(first.done)

        for idx, transition in enumerate(self.n_step_buffer):
            reward_n = reward_n + (self.gamma ** idx) * transition.reward
            next_state_n = transition.next_state
            done_n = transition.done

            reached_n = idx + 1 >= self.n_step
            terminated = bool(transition.done.item() > 0.5) or transition.next_state is None
            if reached_n or terminated:
                break

        if force_terminal:
            next_state_n = None
            done_n = torch.ones_like(done_n)
        return Transition(first.state, first.action, next_state_n, reward_n, done_n)

    def _push_n_step_transition(self, force_terminal: bool = False) -> None:
        transition = self._build_n_step_transition(force_terminal=force_terminal)
        self.memory.push(*transition)
        self.n_step_buffer.popleft()

    def store_transition(self, state, action, next_state, reward, done: bool = False) -> None:
        transition = Transition(
            self._as_tensor(state),
            self._as_action_tensor(action),
            self._as_tensor(next_state),
            self._as_reward_tensor(reward),
            self._as_done_tensor(done),
        )

        if self.n_step == 1:
            self.memory.push(*transition)
            return

        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) >= self.n_step:
            self._push_n_step_transition()
        if bool(done):
            while self.n_step_buffer:
                self._push_n_step_transition()

    def finalize_episode(self, force_terminal: bool = True) -> None:
        if self.n_step == 1:
            return
        while self.n_step_buffer:
            self._push_n_step_transition(force_terminal=force_terminal)

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _current_per_beta(self) -> float:
        if not self.per_enabled:
            return 1.0
        frac = min(1.0, self.optimize_steps / float(self.per_beta_steps))
        return self.per_beta_start + frac * (1.0 - self.per_beta_start)

    def _optimize_impl(self, use_double_dqn: bool) -> Optional[Tuple[float, float]]:
        if len(self.memory) < self.batch_size:
            return None
        if self.use_noisy:
            self.policy_net.train()
            self._reset_policy_noise()

        per_indices = None
        per_weights = torch.ones((self.batch_size, 1), device=self.device, dtype=torch.float32)
        if self.per_enabled:
            beta = self._current_per_beta()
            transitions, per_indices, weights = self.memory.sample(self.batch_size, beta=beta)
            per_weights = torch.as_tensor(weights, device=self.device, dtype=torch.float32).view(self.batch_size, 1)
        else:
            transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        non_final_states = []
        non_final_indices = []
        done_batch = torch.cat(batch.done).view(self.batch_size, -1)
        for idx, next_state in enumerate(batch.next_state):
            if next_state is None:
                continue
            if bool(done_batch[idx].item() > 0.5):
                continue
            non_final_states.append(next_state)
            non_final_indices.append(idx)

        non_final_next_states = None
        if non_final_states:
            non_final_next_states = torch.cat(non_final_states).view(len(non_final_states), -1).unsqueeze(1)

        state_batch = torch.cat(batch.state).view(self.batch_size, -1).unsqueeze(1)
        action_batch = torch.cat(batch.action).view(self.batch_size, -1).long()
        reward_batch = torch.cat(batch.reward).view(self.batch_size, -1)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros((self.batch_size, 1), device=self.device)
        if non_final_next_states is not None:
            with torch.no_grad():
                if use_double_dqn:
                    next_state_action = self.policy_net(non_final_next_states).argmax(dim=1, keepdim=True)
                    target_next_q = self.target_net(non_final_next_states).gather(1, next_state_action)
                else:
                    target_next_q = self.target_net(non_final_next_states).max(dim=1, keepdim=True).values
                next_state_values[non_final_indices] = target_next_q

        gamma = self.n_step_discount if self.n_step > 1 else self.gamma
        expected_state_action_values = reward_batch + (1.0 - done_batch) * gamma * next_state_values

        if self.per_enabled:
            loss = weighted_huber_q_loss(state_action_values, expected_state_action_values, per_weights)
            td_errors = (expected_state_action_values - state_action_values).detach().abs().view(-1)
            priorities = td_errors.cpu().numpy() + self.per_eps
            self.memory.update_priorities(per_indices, priorities)
        else:
            loss = mse_q_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.optimize_steps += 1
        avg_q_value = float(state_action_values.detach().mean().item())
        return float(loss.item()), avg_q_value

    def optimize(self) -> Optional[Tuple[float, float]]:
        return self._optimize_impl(use_double_dqn=False)

    def optimize_double_dqn(self) -> Optional[Tuple[float, float]]:
        return self._optimize_impl(use_double_dqn=True)
