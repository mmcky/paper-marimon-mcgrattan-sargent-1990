# Comparison: Our Notebook vs. DeepSeek Notebook

## Replicating Marimon, McGrattan, and Sargent (1990)

This document compares two Jupyter notebook implementations that aim to replicate
the computational experiments from:

> Marimon, R., McGrattan, E., & Sargent, T. J. (1990). "Money as a Medium of
> Exchange in an Economy with Artificially Intelligent Agents." *Journal of
> Economic Dynamics and Control*, 14, 329-373.

- **Our notebook**: `jupyter/marimon_mcgrattan_sargent_1990.ipynb`
- **DeepSeek notebook**: `MMS_deepseek.ipynb` (provided externally)

---

## 1. Faithfulness to the Paper

**Our notebook** closely follows the paper's actual mechanisms:

- **Trinary encoding** with proper 2-bit numeric representation (`[1,0]`, `[0,1]`,
  `[0,0]`) and `#` (represented as `-1`) wildcards, matching Table 1 in the paper
- **Bucket brigade** strength updates using the paper's equations (10–11):
  cumulative average with inter-classifier payments between exchange and
  consumption systems
- **Genetic algorithm** with two-point crossover, generalization operators, and
  frequency $f(t) = 1/\sqrt{t}$ as specified in the paper
- **Complete enumeration** (72 trade + 12 consume classifiers) for Economies
  A1/A2/B, random initialization for A1.2 and C — matching the paper's two
  approaches
- Reproduces **all five economies**: A1.1, A1.2, A2.1, B.1, and C (fiat money)

**DeepSeek's notebook** takes significant shortcuts:

- Uses **string-based encoding** (`"10#01#"`) rather than proper numeric arrays —
  functional but less faithful
- **Strength updates are ad hoc**: `winner.strength = old_strength - bid_amount + reward`
  instead of the paper's cumulative-average bucket brigade (eqs. 10–13)
- The GA is a **rough approximation** — creates variations with random probability
  rather than the paper's structured crossover/mutation/generalization operators
- **Initializes strengths with "rational" priors** (10.0 for correct actions, 1.0
  otherwise) — the paper explicitly starts all strengths at 0 and lets agents
  learn from scratch
- Only simulates **Economies A1 and A2** (100 periods each), missing B and C
  entirely

---

## 2. Scale and Results

| Aspect | Our Notebook | DeepSeek |
|--------|-------------|----------|
| Agents per type | 50 (paper's value) | 10 |
| Periods (A1) | 1,000 | 100 |
| Periods (A1.2/GA) | 2,000 | — |
| Economy B | Yes | No |
| Economy C (fiat money) | Yes | No |
| Classifier system per type | Shared (as in paper) | Per individual agent |

Our notebook uses 50 agents per type and 1,000–2,000 periods as in the paper.
DeepSeek uses only 10 agents and 100 periods, which is too short for meaningful
convergence.

---

## 3. Architecture Decision: Shared vs. Individual Classifiers

**Our notebook**: All agents of the same type share one classifier system (as the
paper specifies — "to economize on computation," Section 5). This means 3
classifier systems total for 150 agents.

**DeepSeek**: Each of the 30 individual agents gets its own classifier system.
This is 30× more classifier systems but doesn't match the paper's design. With
only 10 agents per type and 100 periods, individual classifiers get too few
learning episodes to converge.

---

## 4. Key Mechanism Differences

### Bucket Brigade (strength updates)

- **Ours**: Implements the paper's interconnected payment chain — the consumption
  classifier pays the exchange classifier, which pays the previous consumption
  classifier. Strengths are *cumulative averages* (eqs. 12–13), which is crucial
  for convergence via stochastic approximation.
- **DeepSeek**: Simple `strength = old - bid + reward`. No inter-system payments,
  no cumulative averaging. This fundamentally changes the learning dynamics.

### Auction

- **Ours**: Highest-strength matching classifier wins (as in the paper — strength
  *is* the bid, modulated by specificity)
- **DeepSeek**: Same basic idea, but the bid function multiplies
  `(b11 + b12 * specificity) * strength`, which is closer to Holland's original
  formulation than the paper's modification

### Genetic Algorithm

- **Ours**: Two-point crossover between strength-proportionate selected pairs,
  with separate generalization operator (replace specific bits with `#`) at rate
  $p_g$
- **DeepSeek**: Randomly modifies bits of the winning classifier with 10%
  probability per position — no crossover, no proper selection

---

## 5. Results Quality

**Our notebook** produces results matching the paper:

- **A1.1**: Type 1 holds 100% Good 2, Type 3 holds 100% Good 1 — matches Table 4
- **A2.1**: Fundamental equilibrium despite high $u$ — matches the paper's key
  finding (Section 7.2)
- **Economy C**: Agents learn to accept fiat money

**DeepSeek** acknowledges its own limitations: "Average absolute difference" is
reported but with only 100 periods and 10 agents, the results are noisy and don't
clearly demonstrate convergence.

---

## 6. Presentation and Exposition

### DeepSeek does well in:

- Clean separation of economy setup, classifier system, and simulation engine
  into distinct classes
- Explicit comparison tables against theoretical predictions (the
  `analyze_results` function)
- The `examine_classifier_strengths` function is nicely structured

### Our notebook does well in:

- Rich mathematical exposition connecting code to the paper's equations
- Holdings distribution plots over time (replicating the paper's Figures 5–8)
- Exchange pattern triangle diagram (replicating Figure 2)
- Classifier inspection with human-readable decoding (G1, G2, G3, ¬G1, etc.)
- Strength distribution histograms
- Economy C fiat money simulation with emergence of money as medium of exchange

---

## 7. What We Could Learn from DeepSeek

- The `analyze_results` function with explicit theoretical vs. empirical
  comparison tables is a nice pedagogical touch we could add
- The structured `print` discussion/conclusions format is readable

---

## Summary

Our notebook is a substantially more faithful and complete replication. DeepSeek's
version is more of a *sketch* — it captures the general idea but takes shortcuts
on every core mechanism (strength updates, GA, scale, shared classifiers) that
materially affect whether the results actually replicate the paper's findings.

The 10-agent, 100-period setup with pre-seeded "rational" initial strengths means
the DeepSeek simulation is largely running on its initial conditions rather than
demonstrating emergent learning, which is the entire point of the paper.
