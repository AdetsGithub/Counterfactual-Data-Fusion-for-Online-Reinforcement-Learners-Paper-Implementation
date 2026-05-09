"""Data-fusion RDC agent: cross-intent, cross-arm, and combined fusion estimates.

Usage: agent = DataFusionRDCAgent(); a = agent.choose_action(intent); agent.update(intent, a, reward)
"""

import numpy as np

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

    def cross_intent_estimate(self, query_action, query_intent):
        query_action = self._validate_action(query_action)
        query_intent = self._validate_intent(query_intent)
        p_w = self.datasets.intent_prior[query_intent]
        if p_w <= _MIN_VARIANCE:
            return float(self.Q_table[query_intent, query_action])
        return self._arm_residual(query_action, query_intent) / p_w

    def _h_xarm(self, query_action, query_intent, alt_arm):
        alt_arm = self._validate_action(alt_arm)
        denominator = self._arm_residual(alt_arm, query_intent)
        if abs(denominator) <= _MIN_VARIANCE:
            return None
        numerator = (
            self._arm_residual(query_action, query_intent)
            * self.Q_table[query_intent, alt_arm]
        )
        return numerator / denominator

    def cross_arm_estimate(self, query_action, query_intent):
        query_action = self._validate_action(query_action)
        query_intent = self._validate_intent(query_intent)
        weighted_sum = 0.0
        weight_sum = 0.0
        for s in range(self.num_arms):
            if s == query_action:
                continue
            h = self._h_xarm(query_action, query_intent, s)
            if h is None or not np.isfinite(h):
                continue
            var = max(self.get_variance(query_intent, s), _MIN_VARIANCE)
            weight = 1.0 / var
            weighted_sum += h * weight
            weight_sum += weight
        if weight_sum > 0.0:
            return weighted_sum / weight_sum
        return self.cross_intent_estimate(query_action, query_intent)
