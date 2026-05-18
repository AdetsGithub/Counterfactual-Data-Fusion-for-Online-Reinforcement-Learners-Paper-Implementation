"""Greedy Casino simulation comparing Standard MAB, RDC, and Data-Fusion RDC agents."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from data_fusion_rdc_agent import DataFusionRDCAgent
from greedy_casino import GreedyCasino
from mabuc_datasets import MABUCDatasets
from rdc_agent import RDCAgent

K = 20


def sample_reward(matrix, intent, action, rng):
    p = matrix[int(action), int(intent)]
    return int(rng.binomial(1, p))


class StandardMAB:
    """Naive epsilon-greedy bandit that ignores intent."""

    def __init__(self, num_arms=4, epsilon=0.1, rng=None):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.rng = rng if rng is not None else np.random.default_rng()
        self.Q_table = np.zeros(num_arms, dtype=np.float64)
        self.N_table = np.zeros(num_arms, dtype=np.int64)

    def _validate_action(self, action):
        action = int(action)
        if not 0 <= action < self.num_arms:
            raise ValueError(
                f"action must be an integer from 0 to {self.num_arms - 1}"
            )
        return action

    def choose_action(self):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_arms))

        best = np.flatnonzero(self.Q_table == self.Q_table.max())
        return int(self.rng.choice(best))

    def update(self, action, reward):
        action = self._validate_action(action)
        self.N_table[action] += 1
        n = self.N_table[action]
        self.Q_table[action] += (reward - self.Q_table[action]) / n


def run_simulation(k, T, seed=42):
    rng = np.random.default_rng(seed)
    env = GreedyCasino(k=k, rng=rng)
    datasets = MABUCDatasets.from_environment(env.matrix, k=k)
    agents = {
        "Standard MAB": StandardMAB(
            num_arms=k, rng=np.random.default_rng(seed)
        ),
        "RDC Agent": RDCAgent(
            num_arms=k, num_intents=k, rng=np.random.default_rng(seed)
        ),
        "Data Fusion RDC": DataFusionRDCAgent(
            datasets=datasets,
            num_arms=k,
            num_intents=k,
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
                reward_cache[action] = sample_reward(
                    env.matrix, intent, action, env.rng
                )
            reward = reward_cache[action]

            if name == "Standard MAB":
                agent.update(action, reward)
            else:
                agent.update(intent, action, reward)

            wins[name] += reward
            histories[name].append(wins[name] / t)

    return {name: np.array(series) for name, series in histories.items()}


def plot_results(histories, output="results.png"):
    T = len(next(iter(histories.values())))
    steps = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, series in histories.items():
        ax.plot(steps, series, label=name)

    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Average Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run Greedy Casino simulation.")
    parser.add_argument("-k", "--k", type=int, default=K, help="Number of arms/intents")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Total steps T (default: 400 * k^2)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output", default="results.png", help="Output plot path"
    )
    args = parser.parse_args()

    T = args.steps if args.steps is not None else int(400 * (args.k ** 2))
    histories = run_simulation(k=args.k, T=T, seed=args.seed)
    plot_results(histories, output=args.output)


if __name__ == "__main__":
    main()
