from __future__ import annotations

import random
from collections import namedtuple
from typing import List, Tuple

import numpy as np


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory: List[Transition] = []
        self.position = 0

    def push(self, *args) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be > 0.")
        if alpha < 0.0:
            raise ValueError("PER alpha must be >= 0.")
        self.capacity = capacity
        self.alpha = alpha
        self.memory: List[Transition] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def push(self, *args) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float) -> Tuple[list[Transition], np.ndarray, np.ndarray]:
        if len(self.memory) < batch_size:
            raise ValueError("Not enough samples in replay buffer.")
        if beta < 0.0:
            raise ValueError("PER beta must be >= 0.")

        probs = self._probabilities()
        indices = np.random.choice(len(self.memory), size=batch_size, replace=True, p=probs)
        samples = [self.memory[int(i)] for i in indices]

        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max() + 1e-8
        return samples, indices.astype(np.int64), weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, priority in zip(indices, priorities):
            value = float(priority)
            if value <= 0.0:
                value = 1e-8
            self.priorities[int(idx)] = value
            if value > self.max_priority:
                self.max_priority = value

    def _probabilities(self) -> np.ndarray:
        valid_priorities = self.priorities[: len(self.memory)]
        scaled = np.power(valid_priorities, self.alpha)
        total = float(np.sum(scaled))
        if not np.isfinite(total) or total <= 0.0:
            return np.full(len(self.memory), 1.0 / len(self.memory), dtype=np.float32)
        return (scaled / total).astype(np.float32)

    def __len__(self) -> int:
        return len(self.memory)
