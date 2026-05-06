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
