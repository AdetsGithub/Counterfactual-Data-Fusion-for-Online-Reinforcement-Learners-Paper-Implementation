"""Greedy Casino simulation comparing Standard MAB, RDC, and Data-Fusion RDC agents."""

import numpy as np

from greedy_casino import GreedyCasino


def sample_reward(intent, action, rng):
    p = GreedyCasino.PAYOUT_MATRIX[int(action), int(intent)]
    return int(rng.binomial(1, p))


class StandardMAB:
    """Naive epsilon-greedy bandit that ignores intent."""

    def __init__(self, num_arms=4, epsilon=0.1, rng=None):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.rng = rng if rng is not None else np.random.default_rng()
        self.Q_table = np.zeros(num_arms, dtype=np.float64)
        self.N_table = np.zeros(num_arms, dtype=np.int64)

    def _validate_action(self, action):
        action = int(action)
        if not 0 <= action < self.num_arms:
            raise ValueError(
                f"action must be an integer from 0 to {self.num_arms - 1}"
            )
        return action

    def choose_action(self):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_arms))

        best = np.flatnonzero(self.Q_table == self.Q_table.max())
        return int(self.rng.choice(best))

    def update(self, action, reward):
        action = self._validate_action(action)
        self.N_table[action] += 1
        n = self.N_table[action]
        self.Q_table[action] += (reward - self.Q_table[action]) / n
