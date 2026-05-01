"""Greedy Casino MABUC environment.

Usage: env = GreedyCasino(); i = env.get_intent(); r = env.pull_arm(0)
"""

import numpy as np


class GreedyCasino:
    PAYOUT_MATRIX = np.array(
        [
            [0.20, 0.30, 0.50, 0.60],
            [0.60, 0.20, 0.30, 0.50],
            [0.50, 0.60, 0.20, 0.30],
            [0.30, 0.50, 0.60, 0.20],
        ],
        dtype=np.float64,
    )

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self._current_intent = None
