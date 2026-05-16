"""Greedy Casino MABUC environment.

Usage: env = GreedyCasino(k=20); i = env.get_intent(); r = env.pull_arm(0)
"""

import numpy as np


class GreedyCasino:
    def __init__(self, k, rng=None):
        self.k = int(k)
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        self.rng = rng if rng is not None else np.random.default_rng()
        self.matrix = self.rng.uniform(0.40, 0.80, size=(self.k, self.k)).astype(
            np.float64
        )
        np.fill_diagonal(self.matrix, 0.20)
        self._current_intent = None

    def get_intent(self):
        self._current_intent = int(self.rng.integers(self.k))
        return self._current_intent

    def pull_arm(self, action):
        if self._current_intent is None:
            raise RuntimeError("call get_intent() before pull_arm()")

        action = int(action)
        if not 0 <= action < self.k:
            raise ValueError(
                f"action must be an integer from 0 to {self.k - 1}"
            )

        p_win = self.matrix[action, self._current_intent]
        return int(self.rng.binomial(1, p_win))
