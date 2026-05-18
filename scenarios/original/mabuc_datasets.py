"""Initial observational and experimental priors for Greedy Casino MABUC.

See THE_CASINO.md, "Observational vs. Experimental Data" (Table b).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MABUCDatasets:
    """Initial observational and experimental priors for Greedy Casino MABUC.

    Experimental values E[Y_x] equal the row-mean of GreedyCasino.PAYOUT_MATRIX
    under uniform intent (0.25 * (0.20 + 0.30 + 0.50 + 0.60) = 0.40).
    """

    observational: np.ndarray  # E[Y|X] - natural win rate per arm
    experimental: np.ndarray  # E[Y_x] - randomized trial win rate per arm
    intent_prior: np.ndarray  # P(I) - probability of each intent profile

    def __post_init__(self):
        for name, arr in (
            ("observational", self.observational),
            ("experimental", self.experimental),
            ("intent_prior", self.intent_prior),
        ):
            if arr.shape != (4,):
                raise ValueError(f"{name} must have shape (4,), got {arr.shape}")
        if not np.isclose(self.intent_prior.sum(), 1.0):
            raise ValueError("intent_prior must sum to 1.0")

    @classmethod
    def greedy_casino(cls) -> "MABUCDatasets":
        return cls(
            observational=np.full(4, 0.20, dtype=np.float64),
            experimental=np.full(4, 0.40, dtype=np.float64),
            intent_prior=np.full(4, 0.25, dtype=np.float64),
        )


GREEDY_CASINO_DATASETS = MABUCDatasets.greedy_casino()
