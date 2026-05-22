## RDCAgent

### SCMs and Counterfactuals

We take the conceptual problem of hidden influences (like the blinking lights or a gambler's sobriety from the Greedy Casino example) and translate it into an SCM.

The core insight introduced here is the concept of the agent's **Intent** - the action an agent naturally wants to take before applying a conscious strategy. By capturing this natural urge, the agent can use it as a mathematical proxy to detect those hidden confounders and adapt its strategy accordingly. 


We have two types of variables:

- **Endogenous ($V$):** Things we can directly observe (e.g., the machine a gambler chooses, the resulting payout).

- **Exogenous ($U$):** Hidden, unobserved variables (e.g., the gambler's sobriety, the blinking lights).

This SCM blueprint generates three distinct "worlds" of probability:

1. **The Observational World ($P(V=v)$):** This represents natural, unbothered behavior. Gamblers walk in, their hidden traits ($U$) influence their machine choice naturally, and we record the results.

2. **The Experimental World ($P(Y | do(X=x))$):** This introduces the **$do()$ operator**, representing a physical intervention. If the investigator _forces_ a gambler to play Machine 0 ($do(X=0)$), they literally sever the causal link between the gambler's hidden traits and their machine choice.

3. **The Counterfactual World ($Y_x(u) = y$):** It asks a retrospective "what if" question about a specific, past event. For example: "Given that a specific drunk gambler naturally played Machine 3 and lost (where $u$ represents their specific hidden state), what _would have happened_ to that exact same gambler if the machine had been 1 instead?"

**Confounding bias** exists whenever the natural world doesn't match the forced experimental world, written mathematically as $P(Y | do(X=x)) \neq P(Y | X=x)$.

### The MABUC Framework

MABUC stands for **M**ulti-**A**rmed **B**andit with **U**nobserved **C**onfounders. 
- A "Multi-Armed Bandit" is just math-speak for a trial-and-error problem where an agent tries to maximize rewards by choosing between different options (like pulling different slot machine arms). MABUC is this classic problem, but complicated by hidden variables.

Let's walk through exactly what happens in a single round (time step $t$) by mapping Definition 3.3 directly to the nodes in the causal diagram:

```tikz
\usetikzlibrary{positioning, arrows.meta, shapes.geometric}

\begin{document}
\begin{tikzpicture}[
    node distance=1.5cm and 1.5cm,
    % Style for observed variables (solid blue circles)
    observed/.style={
        circle, 
        fill=blue!70!black!60, % Adjusted to match the muted blue in the image
        draw=black, 
        minimum size=0.6cm,
        inner sep=0pt,
        font=\bfseries
    },
    % Style for the unobserved variable (white circle)
    unobserved/.style={
        circle, 
        fill=white, 
        draw=black, 
        minimum size=0.6cm,
        inner sep=0pt,
        font=\bfseries
    },
    % Style for the square node (pi_t)
    policy/.style={
        rectangle, 
        fill=blue!70!black!60, 
        draw=black, 
        minimum size=0.5cm,
        inner sep=0pt
    },
    % Arrow styling
    arrow/.style={
        ->, 
        >=Stealth, 
        thick
    },
    % Dashed arrow styling for unobserved effects
    dashedarrow/.style={
        ->, 
        >=Stealth, 
        thick, 
        dashed
    }
]

    % --- Nodes ---
    % Central column
    \node[observed] (It) {$I_t$};
    \node[policy, below=0.8cm of It] (pit) {};
    \node[right=0.0cm of pit] {$\pi_t$}; % Label next to square
    \node[observed, below=0.8cm of pit] (Xt) {$X_t$};
    
    % Left node
    \node[observed, left=of pit] (Ht) {$H_t$};
    
    % Right node
    \node[observed, right=1.5cm of Xt] (Yt) {$Y_t$};
    
    % Top unobserved node
    \node[unobserved, above right=0.8cm and 0.8cm of It] (Ut) {$U_t$};

    % --- Edges ---
    % Solid paths
    \draw[arrow] (Ht) -- (pit);
    \draw[arrow] (It) -- (pit);
    \draw[arrow] (pit) -- (Xt);
    \draw[arrow] (Xt) -- (Yt);
    
    % Dashed paths from Ut
    \draw[dashedarrow] (Ut) -- (It);
    \draw[dashedarrow] (Ut) -- (Yt);

\end{tikzpicture}
\end{document}
```


- **$U_t$ (Unobserved Confounder):** This is the hidden state (like whether the gambler is drunk). 

- **$I_t$ (Intent):** This is the agent's natural "urge" or the action they _would_ take naturally. Notice the dashed arrow from $U_t \rightarrow I_t$. The hidden state is what creates this intent.

- **$H_t$ (History):** The agent's memory of all past intents, choices, and rewards.

- **$\pi_t$ (Policy):** This is the algorithm's brain. It's drawn as a square because it represents a calculated _decision point_. It takes what it remembers ($H_t$) and its current natural urge ($I_t$) and decides what to actually do.

- **$X_t$ (Choice):** The final, deliberate choice of which machine to play based on the policy.

- **$Y_t$ (Reward):** The payout (win/loss). The reward is dictated by the machine chosen ($X_t$) _and_ the hidden state of the world ($U_t$).

Look closely at the policy node ($\pi_t$) in the diagram. It has arrows pointing to it from History ($H_t$) and Intent ($I_t$), but no arrow from the hidden confounders ($U_t$).

### Intent and the Regret Decision Criterion

To outsmart the casino, the algorithm leverages **Definition 3.2 (Intent)**.

Because the unobserved confounders ($U_t$, like drunkenness and blinking lights) are invisible, the agent can't measure them. But it _can_ measure what those confounders make the gambler want to do. **Intent ($I_t$) acts as a perfect mathematical proxy for the hidden state.** If a gambler's natural urge is to walk up to Machine 3, the algorithm doesn't need to know _why_. It just knows that "Intent = Machine 3" corresponds to a specific, vulnerable state that the casino is trying to exploit.

This leads directly to the ultimate strategy: **Definition 3.4 (The Regret Decision Criterion, or RDC)**.

The RDC provides a formal mathematical rule for how the agent should make its final choice:

$$\arg\max_a E[Y_{X=a} | X = i]$$

1. **$X = i$ (Condition on Intent):** First, acknowledge your natural urge. The algorithm asks, "Which machine do I currently _want_ to play?" (Let's say the intent $i$ is Machine 3).

2. **$Y_{X=a}$ (The Counterfactual):** Now, pause. Ask the counterfactual question: "Given that I am currently in a state where I want to play Machine 3, what would my payout ($Y$) be if I _actually_ forced myself to pull arm $a$ instead?"

3. **$\arg\max_a$ (Maximize):** Calculate this expected counterfactual payout for _every_ machine ($a = 0, 1, 2, 3$). Then, pull the arm ($a$) that gives you the highest mathematical expectation.

#### The Policy ($\pi$) as the Counterfactual Engine

The policy $\pi_t$ is the active execution of the Regret Decision Criterion (RDC). It doesn't just mindlessly react; it calculates. Every time a new round starts:

- It acknowledges the current intent $i$.
- It runs the counterfactual "what-if" scenario for every possible action $a$ to find the expected reward.
- It executes the $\arg\max$ operator to deliberately select the action with the highest expected payout, completely overriding the naive urge if a better option exists.

#### History ($H_t$) as a Q-Table

$H_{t}$ can be understood as a Q-table. In standard Reinforcement Learning, a Q-table stores the expected reward for taking a specific action in a specific state, usually written as $Q(s, a)$.

In the MABUC framework, because the agent cannot see the true state of the world ($U_t$), it uses its own intent ($I_t$) as the state proxy. Therefore, the history $H_t$ can be perfectly conceptualized as a Q-table that maps and stores these counterfactual expectations:

$$Q(i, a) \approx E[Y_{X=a} | X=i]$$

If we visualize $H_t$ as a Q-table for the Greedy Casino:

- **The Rows (States):** The agent's intents (e.g., Intent = Machine 0, Intent = Machine 1).    
- **The Columns (Actions):** The machine the agent _actually_ decides to play (Play 0, Play 1, Play 2, Play 3).
- **The Values (Expectations):** The historical win rates the agent has learned over time by tracking what happens when it follows versus defies its intent.

When the agent explores, it updates the values in this Q-table. Once the table has enough data, the policy $\pi_t$ simply looks up the row matching its current intent, scans across the columns, and picks the action with the highest historical payout.

### Putting it Together: Beating the Casino

Imagine our gambler is drunk and the lights are blinking. Their **Intent ($I_t$)** is to play Machine 3. The casino knows this, and if they play Machine 3, their payout is rigged to be **20%**.

A standard randomized experiment (like the investigator) ignores intent entirely, picks a machine at random, and gets an average **40%** win rate.

But the **RDC Agent** says: _"Wait, my intent is Machine 3. That means I am currently in the 'Drunk/Blinking' hidden state. Let me look at the payout matrix for this specific state."_

- If I obey my intent and play Machine 3, I win 20%.
- If I defy my intent and play Machine 0, I win 60%.
- If I defy my intent and play Machine 1, I win 50%.
- If I defy my intent and play Machine 2, I win 30%.

The $\arg\max$ operation tells the agent to pick the maximum value. So, the agent deliberately chooses **Machine 0**, securing a **60% win rate** - beating both the natural gamblers (20%) and the random experimenters (40%).

In summary, at every round $t$ of MABUC, the unobserved state $u_{t}$ is drawn from $P (u)$, which then decides it, which is then considered by the strategy $\pi_{t}$ in concert with the game’s history $h_{t}$; the strategy makes a final arm choice, which is then pulled, as represented by $x_{t}$, and the reward $y_{t}$ is revealed.