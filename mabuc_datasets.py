"""Initial observational and experimental priors for Greedy Casino MABUC.

See THE_CASINO.md, "Observational vs. Experimental Data" (Table b).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MABUCDatasets:
    """Initial observational and experimental priors for Greedy Casino MABUC."""

    observational: np.ndarray  # E[Y|X] — natural win rate per arm
    experimental: np.ndarray  # E[Y_x] — randomized trial win rate per arm
    intent_prior: np.ndarray  # P(I) — probability of each intent profile
