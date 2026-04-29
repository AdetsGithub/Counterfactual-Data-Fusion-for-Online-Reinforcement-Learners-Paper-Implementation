# Counterfactual Data-Fusion for Online Reinforcement Learners

This repository contains a complete Python implementation of the paper **Counterfactual Data-Fusion for Online Reinforcement Learners**.

It demonstrates how Reinforcement Learning agents can use causal inference, counterfactual reasoning, and data fusion to detect and outsmart hidden confounders in their environment, drastically improving learning speed and total rewards.

## The Greedy Casino Problem

Imagine a casino with four new slot machines. The casino uses hidden sensors to detect if a gambler is drunk or sober, and knows whether the machines' lights are blinking or not. These are **Unobserved Confounders (UCs)** to the gambler. 

The casino knows that gamblers' natural machine choices are perfectly predictable based on these UCs. They rig the payouts so that if a gambler follows their "natural intent," they will only win 20% of the time. If an investigator forces gamblers to play randomly, the win rate is 40%. 

To beat the casino, an agent cannot just play randomly. It must use the **Regret Decision Criterion (RDC)**: acknowledging its own natural intent as a proxy for the hidden confounders, and actively choosing the counterfactual action with the highest expected payout. This optimal strategy yields a 60% win rate.

## Features

* **Greedy Casino Environment:** A custom, lightweight MAB environment that generates hidden states and calculates natural intents without relying on heavy RL frameworks.
* **Standard RDC Agent:** An agent that learns to beat the casino from scratch using counterfactual reasoning.
* **Data-Fusion RDC Agent:** An advanced agent that mathematically fuses prior Observational (20% win rate) and Experimental (40% win rate) datasets to deduce counterfactual payouts instantly.
* **Three Fusion Strategies Implemented:** 
    * *Cross-Intent Learning*: Deducing payouts across different hidden states.
    * *Cross-Arm Learning*: Deducing payouts across different machines using inverse-variance weighting.
    * *Combined Approach*: Fusing all available data for maximum statistical robustness.


## References

This implementation is based on the theoretical frameworks of Structural Causal Models (SCMs) and Multi-Armed Bandits with Unobserved Confounders (MABUC).

* *Counterfactual Data-Fusion for Online Reinforcement Learners* (Bareinboim, Forney, Pearl).
```
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