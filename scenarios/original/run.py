"""Run the original 4x4 Greedy Casino simulation."""

import numpy as np

from core.data_fusion_rdc_agent import DataFusionRDCAgent
from core.rdc_agent import RDCAgent
from core.standard_mab import StandardMAB
from scenarios.original.greedy_casino import GreedyCasino
from scenarios.original.mabuc_datasets import GREEDY_CASINO_DATASETS


def sample_reward(intent, action, rng):
    p = GreedyCasino.PAYOUT_MATRIX[int(action), int(intent)]
    return int(rng.binomial(1, p))


def run_simulation(T=50_000, seed=42):
    rng = np.random.default_rng(seed)
    env = GreedyCasino(rng=rng)
    agents = {
        "Standard MAB": StandardMAB(rng=np.random.default_rng(seed)),
        "RDC Agent": RDCAgent(rng=np.random.default_rng(seed)),
        "Data Fusion RDC": DataFusionRDCAgent(
            datasets=GREEDY_CASINO_DATASETS,
            rng=np.random.default_rng(seed),
        ),
    }
    wins = {name: 0 for name in agents}
    histories = {name: [] for name in agents}

    for t in range(1, T + 1):
        intent = env.get_intent()
        reward_cache = {}
        actions = {}

        for name, agent in agents.items():
            if name == "Standard MAB":
                actions[name] = agent.choose_action()
            else:
                actions[name] = agent.choose_action(intent)

        for name, agent in agents.items():
            action = actions[name]
            if action not in reward_cache:
                reward_cache[action] = sample_reward(intent, action, env.rng)
            reward = reward_cache[action]

            if name == "Standard MAB":
                agent.update(action, reward)
            else:
                agent.update(intent, action, reward)

            wins[name] += reward
            histories[name].append(wins[name] / t)

    return {name: np.array(series) for name, series in histories.items()}
