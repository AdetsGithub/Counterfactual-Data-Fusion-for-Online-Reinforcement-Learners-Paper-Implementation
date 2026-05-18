## The Greedy Casino

The scenario illustrates how a system (the casino) can use hidden, confounding variables to manipulate users and bypass regulations, creating a mathematical paradox where observational data and experimental data tell completely completely different stories.

### The Setup: The Gambler's Choice

The casino introduces four new themed slot machines ($X \in \{0, 1, 2, 3\}$).

Through testing, the casino discovers that a gambler's choice of machine is entirely predictable based on two binary UCs:

- **$B \in \{0, 1\}$:** Whether the machines are blinking ($1$) or not ($0$). The casino sets the machines to blink exactly 50% of the time, so $P(B = 1) = 0.5$.

- **$D \in \{0, 1\}$:** Whether the gambler is drunk ($1$) or not ($0$). The casino knows their patrons are drunk 50% of the time, so $P(D = 1) = 0.5$.

A team of psychologists determines that the gambler's "natural" machine choice is dictated by a strict structural equation:

$$X \leftarrow f_X(B, D) = B + 2 \cdot D$$

Because both $B$ and $D$ have a 50/50 chance of occurring, the gambler population falls equally into four distinct profiles, perfectly mapping to the four machines:

- Sober & Not Blinking ($D=0, B=0$): Gambler chooses Machine $0$.
    
- Sober & Blinking ($D=0, B=1$): Gambler chooses Machine $1$.
    
- Drunk & Not Blinking ($D=1, B=0$): Gambler chooses Machine $2$.
    
- Drunk & Blinking ($D=1, B=1$): Gambler chooses Machine $3$.
    

### The Legal Loophole & The Payout Matrix

A new law states that slot machines must have a minimum win rate of 30%. The greedy casino wants to maximize profits while technically skirting this law.

They equip the machines with perfect sensors to detect if a gambler is drunk ($D$) and they already know if the machines are blinking ($B$). They program a **reactive payout strategy** based on these variables, resulting in the following:

| **(a)**             | **D=0** | **D=0**     | **D=1**     | **D=1**     |
| ------------------- | ------- | ----------- | ----------- | ----------- |
| $E[y_1 \| X, B, D]$ | $B=0$   | **$B = 0$** | **$B = 1$** | **$B = 0$** |
| $X = 0$             | \*0.20  | 0.30        | 0.50        | 0.60        |
| $X = 1$             | 0.60    | \*0.20      | 0.30        | 0.50        |
| $X = 2$             | 0.50    | 0.60        | \*0.20      | 0.30        |
| $X = 3$             | 0.30    | 0.50        | 0.60        | \*0.20      |



The asterisks (\*) represent the gambler's natural machine choice based on the structural equation above. Notice a pattern?

- If you are sober and the machines aren't blinking, you naturally choose $X=0$. The casino detects this and sets the payout for $X=0$ to **20%** ($0.20$).

- If you are drunk and the machines are blinking, you naturally choose $X=3$. The casino detects this and sets the payout for $X=3$ to **20%** ($0.20$).    

No matter who the gambler is, their natural predilection leads them directly into a trap where their specific machine is rigged to pay out exactly 20% - which is 10% _below_ the legal minimum.

### The Paradox: Observational vs. Experimental Data

Gamblers complain they are only winning 20% of the time. The state sends an investigator. This creates two distinct datasets:

| (b)     | $E[y_1 \| X]$ | $E[y_1 \| do(X)]$ |
| :------ | :-----------: | :---------------: |
| $X = 0$ |     0.20      |       0.40        |
| $X = 1$ |     0.20      |       0.40        |
| $X = 2$ |     0.20      |       0.40        |
| $X = 3$ |     0.20      |       0.40        |

#### The Observational Data ($E[y_1 | X]$)

This is the data collected from the casino floor as gamblers behave naturally. Because of the trap described above, the expected payout of winning ($Y=y_1$) given a natural choice ($X$) is always 20%. Therefore, $E[y_1 | X] = 0.20$ for all machines.

#### The Experimental Data ($E[y_1 | do(X)]$)

The investigator runs a Randomized Control Trial. They force random gamblers to play randomly assigned machines, severing the link between the gambler's traits ($B, D$) and the machine they choose. This is denoted by the causal operator $do(X)$.

Let's do the math on what the investigator sees if they force gamblers to play Machine 0 ($do(X=0)$). The gambler playing could randomly be any of the four profiles (each with a 25% probability). Based on Table (a), the payouts for Machine 0 are:

- 25% chance of $D=0, B=0$: Payout is 0.20
- 25% chance of $D=0, B=1$: Payout is 0.30
- 25% chance of $D=1, B=0$: Payout is 0.50
- 25% chance of $D=1, B=1$: Payout is 0.60

Expected Value $= 0.25(0.20 + 0.30 + 0.50 + 0.60) = 0.25(1.60) = \mathbf{0.40}$

The investigator finds an average win rate of 40% across all machines! The casino is declared innocent, even though natural gamblers are being fleeced at 20%.