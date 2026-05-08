"""Data-fusion RDC agent: cross-intent, cross-arm, and combined fusion estimates.

Usage: agent = DataFusionRDCAgent(); a = agent.choose_action(intent); agent.update(intent, a, reward)
"""

from mabuc_datasets import GREEDY_CASINO_DATASETS, MABUCDatasets
from rdc_agent import RDCAgent

_MIN_VARIANCE = 1e-12


class DataFusionRDCAgent(RDCAgent):
    def __init__(self, datasets: MABUCDatasets = GREEDY_CASINO_DATASETS, **kwargs):
        super().__init__(**kwargs)
        self.datasets = datasets

    def get_variance(self, intent, action):
        intent = self._validate_intent(intent)
        action = self._validate_action(action)
        n = self.N_table[intent, action]
        if n < 2:
            return 1.0
        p = self.Q_table[intent, action]
        return p * (1.0 - p)

    def _arm_residual(self, query_action, query_intent):
        query_action = self._validate_action(query_action)
        query_intent = self._validate_intent(query_intent)
        residual = self.datasets.experimental[query_action]
        for i in range(self.num_intents):
            if i == query_intent:
                continue
            residual -= self.Q_table[i, query_action] * self.datasets.intent_prior[i]
        return residual
