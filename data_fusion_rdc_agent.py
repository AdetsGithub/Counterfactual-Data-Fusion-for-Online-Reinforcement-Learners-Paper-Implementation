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
            return 1.0 # High uncertainty for cold start
            
        p = self.Q_table[intent, action]
        # Clamp probability to prevent variance collapsing to 0
        p_smoothed = max(0.05, min(0.95, p))
        
        # Divide by n to calculate the variance of the sample mean
        return (p_smoothed * (1.0 - p_smoothed)) / n

    def _arm_residual(self, query_action, query_intent):
        query_action = self._validate_action(query_action)
        query_intent = self._validate_intent(query_intent)
        residual = self.datasets.experimental[query_action]
        
        for i in range(self.num_intents):
            if i == query_intent:
                continue
            # If unexplored, substitute the experimental dataset average
            if self.N_table[i, query_action] < 1:
                est = self.datasets.experimental[query_action]
            else:
                est = self.Q_table[i, query_action]
                
            residual -= est * self.datasets.intent_prior[i]
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
            
        # If the alternate arm is unexplored, substitute the experimental average
        if self.N_table[query_intent, alt_arm] < 1:
            alt_est = self.datasets.experimental[alt_arm]
        else:
            alt_est = self.Q_table[query_intent, alt_arm]
            
        numerator = (
            self._arm_residual(query_action, query_intent) * alt_est
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

    def _sigma2_xint(self, query_action, query_intent):
        query_action = self._validate_action(query_action)
        query_intent = self._validate_intent(query_intent)
        total = 0.0
        count = 0
        for i in range(self.num_intents):
            if i == query_intent:
                continue
            total += self.get_variance(i, query_action)
            count += 1
        if count == 0:
            return 1.0
        return total / count

    def _sigma2_xarm(self, query_action, query_intent):
        query_action = self._validate_action(query_action)
        query_intent = self._validate_intent(query_intent)
        k = self.num_intents
        sum_r = sum(
            self.get_variance(i, query_action)
            for i in range(k)
            if i != query_intent
        )
        terms = []
        for s in range(self.num_arms):
            if s == query_action:
                continue
            sum_s = sum(
                self.get_variance(i, s) for i in range(k) if i != query_intent
            )
            var_sw = self.get_variance(query_intent, s)
            terms.append((sum_r + sum_s + var_sw) / (2 * k - 1))
        if not terms:
            return 1.0
        return sum(terms) / len(terms)

    def get_fused_estimate(self, intent, action):
        intent = self._validate_intent(intent)
        action = self._validate_action(action)
        e_samp = self.Q_table[intent, action]
        var_samp = max(self.get_variance(intent, action), _MIN_VARIANCE)
        e_xint = self.cross_intent_estimate(action, intent)
        var_xint = max(self._sigma2_xint(action, intent), _MIN_VARIANCE)
        e_xarm = self.cross_arm_estimate(action, intent)
        var_xarm = max(self._sigma2_xarm(action, intent), _MIN_VARIANCE)
        alpha = e_samp / var_samp + e_xint / var_xint + e_xarm / var_xarm
        beta = 1.0 / var_samp + 1.0 / var_xint + 1.0 / var_xarm
        return alpha / beta

    def choose_action(self, intent):
        intent = self._validate_intent(intent)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_arms))

        row = np.array(
            [self.get_fused_estimate(intent, a) for a in range(self.num_arms)]
        )
        best = np.flatnonzero(row == row.max())
        return int(self.rng.choice(best))
