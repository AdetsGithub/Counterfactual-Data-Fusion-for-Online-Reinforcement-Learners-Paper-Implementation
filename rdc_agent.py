"""Standard Regret Decision Criterion agent (no data fusion).

Usage: agent = RDCAgent(); a = agent.choose_action(intent); agent.update(intent, a, reward)
"""

import numpy as np


class RDCAgent:
    def __init__(self, num_arms=4, num_intents=4, epsilon=0.1, rng=None):
        self.num_arms = num_arms
        self.num_intents = num_intents
        self.epsilon = epsilon
        self.rng = rng if rng is not None else np.random.default_rng()
        self.Q_table = np.zeros((num_intents, num_arms), dtype=np.float64)
        self.N_table = np.zeros((num_intents, num_arms), dtype=np.int64)

    def _validate_intent(self, intent):
        intent = int(intent)
        if not 0 <= intent < self.num_intents:
            raise ValueError(
                f"intent must be an integer from 0 to {self.num_intents - 1}"
            )
        return intent

    def _validate_action(self, action):
        action = int(action)
        if not 0 <= action < self.num_arms:
            raise ValueError(
                f"action must be an integer from 0 to {self.num_arms - 1}"
            )
        return action

    def choose_action(self, intent):
        intent = self._validate_intent(intent)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_arms))

        row = self.Q_table[intent]
        best = np.flatnonzero(row == row.max())
        return int(self.rng.choice(best))

    def update(self, intent, action, reward):
        intent = self._validate_intent(intent)
        action = self._validate_action(action)
        self.N_table[intent, action] += 1
        n = self.N_table[intent, action]
        self.Q_table[intent, action] += (reward - self.Q_table[intent, action]) / n
