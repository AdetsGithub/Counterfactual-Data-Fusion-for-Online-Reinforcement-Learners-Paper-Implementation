"""Initial observational and experimental priors for Greedy Casino MABUC.

See THE_CASINO.md, "Observational vs. Experimental Data" (Table b).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MABUCDatasets:
    """Initial observational and experimental priors for Greedy Casino MABUC.

    Experimental values E[Y_x] equal the row-mean of the payout matrix
    under uniform intent (mean over intent columns per action row).
    """

    observational: np.ndarray  # E[Y|X] - natural win rate per arm
    experimental: np.ndarray  # E[Y_x] - randomized trial win rate per arm
    intent_prior: np.ndarray  # P(I) - probability of each intent profile

    def __post_init__(self):
        k = len(self.observational)
        for name, arr in (
            ("observational", self.observational),
            ("experimental", self.experimental),
            ("intent_prior", self.intent_prior),
        ):
            if arr.shape != (k,):
                raise ValueError(f"{name} must have shape ({k},), got {arr.shape}")
        if not np.isclose(self.intent_prior.sum(), 1.0):
            raise ValueError("intent_prior must sum to 1.0")

    @classmethod
    def from_environment(cls, env_matrix: np.ndarray, k: int) -> "MABUCDatasets":
        env_matrix = np.asarray(env_matrix, dtype=np.float64)
        if env_matrix.shape != (k, k):
            raise ValueError(
                f"env_matrix must have shape ({k}, {k}), got {env_matrix.shape}"
            )
        return cls(
            observational=np.full(k, 0.20, dtype=np.float64),
            experimental=np.mean(env_matrix, axis=1),
            intent_prior=np.full(k, 1.0 / k, dtype=np.float64),
        )
