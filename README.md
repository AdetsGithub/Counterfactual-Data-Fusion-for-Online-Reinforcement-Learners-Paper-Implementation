# Counterfactual Data-Fusion for Online Reinforcement Learners

A Python implementation of [Counterfactual Data-Fusion for Online Reinforcement Learners](https://proceedings.mlr.press/v70/forney17a.html) (Forney, Pearl, Bareinboim, ICML 2017). The code reproduces the **Greedy Casino** multi-armed bandit with unobserved confounders (MABUC) and compares three learning agents: a naive baseline, an online **Regret Decision Criterion (RDC)** agent, and a **Data-Fusion RDC** agent that fuses observational and experimental priors via counterfactual reasoning.

## Motivation

How can an agent safely learn from the behavior of others when hidden factors are at play?

- Traditionally, RL agents learn through their own trial-and-error.
- They could learn faster by watching others, but **unobserved confounders (UCs)** can make observational data misleading.
- The paper shows how **causal inference** and **counterfactual data fusion** let an agent combine observational logs with experimental trials without falling into confounder traps.

## Core ideas

Causal data is often described using Judea Pearl’s ladder of causation. An agent may have access to three kinds of information:

| **Type** | **What it represents** | **Example (self-driving car)** |
| --- | --- | --- |
| **Observational** | **Seeing:** passively watching others act | Logs of when a human brakes or accelerates, without knowing why |
| **Experimental** | **Doing:** the agent intervenes and observes outcomes | Random brake tests on an empty track |
| **Counterfactual** | **Imagining:** “what if” a different action had been taken in a past situation | Given the human braked and avoided a crash, what would have happened if they had accelerated instead? |

Standard RL relies heavily on experimental data (exploration). Learning only from observational data is fast but dangerous when UCs influence both actions and rewards. The paper argues that **counterfactual reasoning** is the missing piece for safely fusing observational and experimental data.

The fusion workflow, formalized with a **Structural Causal Model (SCM)**:

1. **Observe** - collect or receive observational logs.
2. **Infer (counterfactual)** - estimate hidden-state effects: given action and outcome, how likely is a UC?
3. **Fuse** - when confident, relabel or combine data points with counterfactual deductions to update the policy faster.

When counterfactual uncertainty is high, the agent falls back to targeted exploration (experimental data) only for those situations-combining the speed of observational learning with the safety of selective experimentation.

## The Greedy Casino problem

The flagship environment is a casino with **K slot machines** (arms) and **K intent profiles** (hidden gambler states). The casino observes confounders the gambler cannot see (e.g. sobriety and blinking lights in the original 4×4 setup) and rigs payouts so that **natural choices** win only **20%** of the time, while **randomized play** wins **40%**. An optimal agent that uses **intent** as a proxy for hidden state and applies the **RDC** can reach roughly **60%** win rate.

In the original scenario, intent is deterministic: \(X \leftarrow B + 2D\) with binary \(B, D\), mapping four UC profiles to four machines. The payout matrix is fixed (see [docs/THE_CASINO.md](docs/THE_CASINO.md) for the full derivation and the observational vs. experimental paradox).

## Agents

Each simulation step: the environment samples an **intent**, each agent chooses an **action** (arm), and all agents share the same Bernoulli reward for a given (intent, action) pair. Cumulative average reward is plotted over time.

| Agent | Module | Uses intent? | Data fusion? | Role |
| --- | --- | --- | --- | --- |
| **Standard MAB** | `core/standard_mab.py` | No | No | Epsilon-greedy baseline; ignores confounding |
| **RDC Agent** | `core/rdc_agent.py` | Yes | No | Learns a \(Q\)-table over (intent, action); explores with decaying \(\epsilon\) |
| **Data Fusion RDC** | `core/data_fusion_rdc_agent.py` | Yes | Yes | Extends RDC with fused counterfactual estimates from priors + online samples |

The Data-Fusion agent implements three fusion strategies from the paper (Equations 3, 7-9), combined via inverse-variance weighting:

- **Cross-intent learning** - deduce payouts across intents from experimental averages.
- **Cross-arm learning** - relate arms under the same intent using shared intent priors.
- **Combined approach** - fuse sample, cross-intent, and cross-arm estimates for each \((\text{intent}, \text{action})\) cell.

See [docs/RDCAGENT.md](docs/RDCAGENT.md) and [docs/DATA_FUSION.md](docs/DATA_FUSION.md) for theory and equations.

## Project structure

```
.
├── main.py                      # CLI entry point
├── requirements.txt
├── core/
│   ├── rdc_agent.py             # RDC agent (Q-table, epsilon decay)
│   ├── data_fusion_rdc_agent.py # Fusion estimates + action selection
│   ├── standard_mab.py          # Naive epsilon-greedy baseline
│   └── plotting.py              # Cumulative reward plots
├── scenarios/
│   ├── original/                # Paper 4×4 Greedy Casino
│   │   ├── greedy_casino.py
│   │   ├── mabuc_datasets.py    # Fixed 20% / 40% priors
│   │   └── run.py
│   └── generalized/             # K×K extension
│       ├── greedy_casino.py     # Procedural payout matrix
│       ├── mabuc_datasets.py    # Priors from environment matrix
│       └── run.py
└── docs/
    ├── THE_CASINO.md
    ├── RDCAGENT.md
    └── DATA_FUSION.md
```

## Getting started

### Requirements

- Python 3.8+
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

### Installation

```bash
git clone <repository-url>
cd Counterfactual-Data-Fusion-for-Online-Reinforcement-Learners-Paper-Implementation
pip install -r requirements.txt
```

### Running simulations

`main.py` runs all three agents in parallel and saves a plot of cumulative average reward.

```bash
# Original 4×4 scenario (default T = 50,000)
python main.py --mode original

# Generalized K×K scenario (default T = 400 × k²)
python main.py --mode generalized --k 20

# Custom step count, seed, and output path
python main.py --mode original --steps 10000 --seed 0 --output my_run.png
python main.py --mode generalized -k 10 --steps 50000 --output k10.png
```

| Flag | Description |
| --- | --- |
| `--mode` | **Required.** `original` (fixed 4×4 casino) or `generalized` (K×K) |
| `-k`, `--k` | Number of arms/intents in generalized mode (default: `20`) |
| `--steps` | Total time steps \(T\). Default: `50000` (original), `400 × k²` (generalized) |
| `--seed` | Random seed (default: `42`) |
| `--output` | Path for the saved plot (default: `results.png`) |

### Expected behavior

On the **original** scenario you should see:

- **Standard MAB** - converges toward the misleading observational average (~20%).
- **RDC Agent** - improves over time toward the optimal counterfactual policy (~60%) via online learning.
- **Data Fusion RDC** - reaches high performance much faster by fusing fixed observational/experimental priors with online counterfactual updates.

The **generalized** scenario uses a random \(K \times K\) payout matrix (diagonal “natural” payouts at 0.20, off-diagonal uniform in \([0.40, 0.80]\)) and derives MABUC priors from that matrix. Larger `k` requires more steps (default scales as \(400k^2\)) for stable learning curves.

## Documentation

| Document | Contents |
| --- | --- |
| [docs/THE_CASINO.md](docs/THE_CASINO.md) | Environment setup, payout matrix, observational vs. experimental paradox |
| [docs/RDCAGENT.md](docs/RDCAGENT.md) | SCMs, MABUC framework, RDC action selection |
| [docs/DATA_FUSION.md](docs/DATA_FUSION.md) | Master equation, three fusion strategies, combined estimator |

## Features

- **Greedy Casino environment** - lightweight MABUC simulator (no heavy RL framework).
- **Standard RDC agent** - intent-conditioned Q-learning with epsilon decay.
- **Data-Fusion RDC agent** - cross-intent, cross-arm, and combined fusion with variance-weighted estimates.
- **Two scenarios** - faithful 4×4 reproduction and scalable K×K generalization.
- **Comparison plots** - cumulative average reward for all agents on one chart.

## References

This implementation follows **Structural Causal Models (SCMs)** and **Multi-Armed Bandits with Unobserved Confounders (MABUC)** as in Bareinboim et al. and Forney et al.

```bibtex
@InProceedings{pmlr-v70-forney17a,
  title = 	 {Counterfactual Data-Fusion for Online Reinforcement Learners},
  author =       {Andrew Forney and Judea Pearl and Elias Bareinboim},
  booktitle = 	 {Proceedings of the 34th International Conference on Machine Learning},
  pages = 	 {1156--1164},
  year = 	 {2017},
  editor = 	 {Precup, Doina and Teh, Yee Whye},
  volume = 	 {70},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {06--11 Aug},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v70/forney17a/forney17a.pdf},
  url = 	 {https://proceedings.mlr.press/v70/forney17a.html},
  abstract = 	 {The Multi-Armed Bandit problem with Unobserved Confounders (MABUC) considers decision-making settings where unmeasured variables can influence both the agent’s decisions and received rewards (Bareinboim et al., 2015). Recent findings showed that unobserved confounders (UCs) pose a unique challenge to algorithms based on standard randomization (i.e., experimental data); if UCs are naively averaged out, these algorithms behave sub-optimally, possibly incurring infinite regret. In this paper, we show how counterfactual-based decision-making circumvents these problems and leads to a coherent fusion of observational and experimental data. We then demonstrate this new strategy in an enhanced Thompson Sampling bandit player, and support our findings’ efficacy with extensive simulations.}
}
```

Paper: [Counterfactual Data-Fusion for Online Reinforcement Learners](https://proceedings.mlr.press/v70/forney17a.html)
