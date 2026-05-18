"""Plot cumulative average reward histories."""

import matplotlib.pyplot as plt
import numpy as np


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
