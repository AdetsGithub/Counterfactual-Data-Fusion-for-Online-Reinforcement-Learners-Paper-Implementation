# Counterfactual Data-Fusion for Online Reinforcement Learners

This repository contains a complete Python implementation of the paper **Counterfactual Data-Fusion for Online Reinforcement Learners**.

It demonstrates how Reinforcement Learning agents can use causal inference, counterfactual reasoning, and data fusion to detect and outsmart hidden confounders in their environment, drastically improving learning speed and total rewards.

## Motivation

How can an agent safely learn from the behavior of others when there are hidden factors at play?
- Traditionally, RL agents learn through their own trial-and-error. 
- Could theoretically learn faster by watching others but, hidden variables - called Unobserved Confounders (UCs) - can make this observed data misleading. 
- Author proposes using causal inference to fuse different types of data to speed up learning without falling into the traps set by these hidden variables

## Overview

The differences between observational, experimental, and counterfactual data, and why the third type is the crucial missing piece.

In causal inference data is categorized by _how_ it was generated
- often framed using Judea Pearl’s "Ladder of Causation."

We can look at the three types of data an agent might use to learn:

| **Type**           | **What it represents**                                                                                                | **Example (Self-Driving Car)**                                                                                                                        |
| ------------------ | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Observational**  | **Seeing:** Data collected passively by watching others act in the world.                                             | Watching a human drive. You record when they brake and accelerate, but you don't always know _why_.                                                   |
| **Experimental**   | **Doing:** Data collected by the agent actively taking an action to see what happens.                                 | The agent randomly testing the brakes on an empty track to see how the car reacts.                                                                    |
| **Counterfactual** | **Imagining:** Reasoning about "what if" a different action was taken in a specific, already-observed past situation. | "Given the human _did_ brake and avoided a crash, what _would have_ happened to that exact car in that exact moment if they had accelerated instead?" |

Traditional RL relies heavily on Experimental data (trial-and-error exploration)
- effective but can be slow and, in the real world, dangerous
- would be much faster to learn from Observational data (like massive logs of human drivers). 

However, the paper argues that the agent needs that third level - Counterfactual data - to safely and effectively fuse the first two.

Observational data may be suboptimal and an agent that copies it would inherit all of the data's mistakes.

This paper focuses on Unobserved Confounders (UCs) - even if the human is acting perfectly, they might be basing their decisions on information the RL agent cannot see.
- UCs  influence both the action taken and the final outcome, confounding the data.

This is what the **fusion engine** in the paper addresses - uses counterfactuals to bridge the gap between observational logs and experimental trial-and-error. 

If there exists data points that rely on the UCs, then we can salvage these data points to learn something valuable.

We use an SCM to formalise this process:
1. **Observe:** The agent looks at the observational log.
2. **Infer (Counterfactual):** The agent uses its SCM to calculate the probability of the hidden state. It asks, "Given this action and outcome, what is the probability that this is due to a UC?"
3. **Fuse:** If the probability is high, the agent essentially re-labels that data point. It takes the observation and fuses it with its counterfactual deduction.

This allows the agent to safely extract a useful rule from imperfect data and allows the online RL agent to learn optimal policies faster than traditional methods.

Instead of blindly exploring everything (which is slow and dangerous) or blindly trusting the logs (which falls into the confounder trap), the agent uses a highly targeted approach.

The agent 
- calculates the probability of a hidden confounder using its counterfactual engine. 
- if it can confidently deduce why the observed data looks the way it does, it safely fuses that data and updates its policy without taking any physical risks.
- if the counterfactual math yields high uncertainty - meaning the observed data doesn't make sense with the causal model - the agent flags that specific situation. That is the trigger. It pauses learning from the log and switches to active, trial-and-error exploration (Experimental data) just for that specific scenario to figure out what's really going on.

By only taking risks when the human data is truly confusing, the agent gets the best of both worlds: the speed of observational learning and the safety of experimental verification.

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