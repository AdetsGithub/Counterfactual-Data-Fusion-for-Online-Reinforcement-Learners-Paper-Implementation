## Fusing Datasets

Up to this point, the agent knows _what_ it needs to do (use the Regret Decision Criterion to maximize rewards based on its Intent), but starting from zero and learning via trial-and-error takes a long time.

**Can the agent use the investigator’s experimental data and the casino floor's observational data to mathematically "cheat" and fill in its knowledge gaps faster?**
- Yes
### The Master Equation (Relating the Datasets)

The agent starts with two massive datasets:

1. **Observational:** $E[Y|X]$ (The natural gamblers winning 20%).
2. **Experimental:** $E[Y|do(X)]$ or $E[Y_x]$ (The investigator’s randomized trials winning 40%).

The section introduces **Equation 2**, based on the law of total probability, which bridges these datasets with the counterfactuals the agent actually wants to learn:

$$E[Y_x] = E[Y_x|x_1]P(x_1) + ... + E[Y_x|x_K]P(x_K)$$

Let's dissect this equation:

- **The Left Side ($E[Y_x]$):** This is purely **Experimental Data**. We already know this from the investigator.

- **The Right Side (The matching terms):** When the action matches the intent ($x = x'$), this is just **Observational Data** due to the "consistency axiom" (e.g., $E[Y_3|x_3]$ is just the natural payout for a gambler who wants to play Machine 3 and does).

- **The Right Side (The mismatched terms):** When the action defies the intent ($x \neq x'$), these are the **Counterfactuals** (e.g., what happens when you want to play Machine 3 but force yourself to play Machine 0).


**Theorem 4.1** is a massive breakthrough here. In standard statistics, counterfactuals (specifically the Effect of Treatment on the Treated, or ETT) are usually impossible to estimate empirically from data alone. This theorem proves that because our agent specifically tracks its _Intent_ before acting, any random exploration it does is strictly counterfactual. Therefore, the agent can actively collect data to fill out a matrix, where the diagonals are natural observations and the off-diagonals are counterfactuals.

### The Three Strategies for "Fusing" Data

Because the agent is playing a game in real-time online learning, it doesn't have infinite data. Sometimes its estimates for a specific counterfactual (let's call it our "query": $E[Y_{x_r} | x_w]$, playing arm $r$ when intending $w$) will be noisy.

The paper proposes three strategies to mathematically deduce the value of a single cell in the matrix by looking at the surrounding data.

#### Strategy 1: Cross-Intent Learning

If we look back at Equation 2, it forms a system of algebraic equations. If we want to find the value for one specific counterfactual, we can simply isolate it using algebra.

**Equation 3** does exactly this. It mathematically calculates the payout of pulling arm $r$ under intent $w$ by subtracting the payouts of arm $r$ under _all other intents_ from the overall experimental average.

$$E_{XInt}[Y_{x_r}|x_w] = [E[Y_{x_r}] - \sum_{i \neq w}^{K} E[Y_{x_r}|x_i]P(x_i)]/P(x_w)$$

- **The takeaway:** Information "leaks" across intents. If you know how an arm performs when you are sober, you can mathematically deduce how it might perform when you are drunk, without even pulling it.

#### Strategy 2: Cross-Arm Learning

This strategy looks for relationships between different arms under the _same_ intent.

The derivation (Equations 4, 5, and 6) relies on the fact that the probability of the intent occurring, $P(x_w)$, is a shared constant. By setting equations for two different arms ($x_r$ and $x_s$) equal to each other, the authors derive **Equation 7**.

$$E[Y_{x_r}] = \sum_{i}^K E[Y_{x_r}|x_i]P(x_i)$$

$$E[Y_{x_s}] = \sum_{i}^K E[Y_{x_s}|x_i]P(x_i)$$

$$P(x_w) = \frac{E[Y_{x_r}] - \sum_{i \neq w}^{K} E[Y_{x_r}|x_i]P(x_i)}{E[Y_{x_r}|x_w]}$$

$$= \frac{E[Y_{x_s}] - \sum_{i \neq w}^{K} E[Y_{x_s}|x_i]P(x_i)}{E[Y_{x_s}|x_w]}$$

$$E[Y_{x_r}|x_w] = \frac{[E[Y_{x_r}] - \sum_{i \neq w}^{K} E[Y_{x_r}|x_i]P(x_i)] E[Y_{x_s}|x_w]}{E[Y_{x_s}] - \sum_{i \neq w}^{K} E[Y_{x_s}|x_i]P(x_i)}$$

- **The takeaway:** If you are in a specific intent state (e.g., Drunk/Blinking), you can estimate the payout of Machine 0 by looking at the data you collected from pulling Machine 1 in that exact same state.

- Because some arms are pulled rarely, **Equation 8** applies an "inverse-variance weighted average." This is just a statistical way of saying: "Average all these estimates together, but trust the ones with a lot of data (low variance) more than the ones with a little data (high variance)."

$$E_{XArm}[Y_{x_r}|x_w] = \frac{\sum_{i \neq r}^{K} h_{XArm}(x_r, x_w, x_i) / \sigma^2_{x_i, x_w}}{\sum_{i \neq r}^{K} 1 / \sigma^2_{x_i, x_w}}$$

#### Strategy 3: The Combined Approach

Why choose one strategy when you can use all three?

**Equation 9** creates a master estimate ($E_{combo}$) for any counterfactual.

$$\alpha = E_{samp}[Y_{x_r}|x_w]/\sigma^2_{x_r,x_w} + E_{XInt}[Y_{x_r}|x_w]/\sigma^2_{XInt} + E_{XArm}[Y_{x_r}|x_w]/\sigma^2_{XArm}$$

$$\beta = 1/\sigma^2_{x_r,x_w} + 1/\sigma^2_{XInt} + 1/\sigma^2_{XArm}$$


$$E_{combo}[Y_{x_r}|x_w] = \frac{\alpha}{\beta}$$
It takes:

1. **$E_{samp}$**: The actual data the agent collected by pulling the arm.

2. **$E_{XInt}$**: The mathematical deduction from Strategy 1 (Cross-Intent).

3. **$E_{XArm}$**: The mathematical deduction from Strategy 2 (Cross-Arm).

It weights them all by their inverse variances (trusting the most statistically stable numbers) to create a highly robust, fused estimate. This allows the agent to learn the optimal winning strategy drastically faster than standard reinforcement learning.

### 3. The Full Pipeline

1. **Initialization:** The agent enters the casino armed with the observational (20%) and experimental (40%) datasets.

2. **The Hidden World:** Unobserved confounders (drunkenness, lights) manifest in the environment.

3. **Intent & Fusion:** The hidden world generates an urge/Intent in the agent. The agent then runs **Strategy 3** to mathematically fuse its historical data and deduce the best counterfactual choice.

4. **Action:** The agent deliberately pulls the mathematically superior arm, ignoring its natural urge.

5. **Observation:** The agent receives a reward (e.g., a 60% win rate), records this highly valuable counterfactual data point, and updates its history for the next round.