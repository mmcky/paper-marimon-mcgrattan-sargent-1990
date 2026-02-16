---
title: "Money as a Medium of Exchange in an Economy with Artificially Intelligent Agents"
---

# Replication of Marimon, McGrattan & Sargent (1990)

This project provides a modern MyST Markdown version of the paper and a Python replication of the classifier system simulations from:

> Marimon, R., McGrattan, E., & Sargent, T. J. (1990). **Money as a Medium of Exchange in an Economy with Artificially Intelligent Agents.** *Journal of Economic Dynamics and Control*, 14(2), 329–373. [DOI: 10.1016/0165-1889(90)90025-C](https://doi.org/10.1016/0165-1889(90)90025-C)

The paper studies how artificially intelligent agents using Holland's classifier systems learn to use commodity money in a Kiyotaki-Wright economy. Agents are endowed with production technologies and must trade to obtain their desired consumption goods. Through a bucket brigade payment system and genetic algorithms, agents learn which goods to accept in trade — and commodity money emerges endogenously.

## The Paper

The [original PDF](../original/Marimon_McGrattan_Sargent_1990.pdf) has been converted to a [MyST Markdown version](../paper/paper.md) — a modern, web-native format with cross-references, proper equation numbering, and navigable structure. The conversion captures ~93% of the original content including all 17 equations, 66 tables, 10 figures, 17 footnotes, and 21 references.

The key ingredients of the paper are:

- **Kiyotaki-Wright (1989) economy**: Multiple agent types with different production technologies, indivisible goods, and random pairwise matching
- **Holland classifier systems**: Trinary-encoded rules (0, 1, #) that compete via an auction mechanism to make trade and consumption decisions
- **Bucket brigade**: A payment system (Equations 10–11) that propagates rewards backward through the chain of classifiers responsible for successful outcomes
- **Genetic algorithm**: Periodic evolution of classifier populations through crossover and mutation to discover better trading strategies

The paper simulates 8 economies with varying parameters and demonstrates convergence to Nash-Markov equilibria, including the emergence of fiat money in Economy C.

**[Read the full paper text →](../paper/paper.md)** | **[PDF → MyST conversion notes →](../paper/2026-02-16-pdf-to-myst-comparison.md)**

## Python Replication

The primary companion notebook replicates all 8 economies from the paper using Python (NumPy, Matplotlib):

:::{card} Companion Notebook 1: Classifier Systems
:link: /jupyter/companion-notebook-1.ipynb

Full replication of all **8 economies** (A1.1, A1.2, A2.1, A2.2, B.1, B.2, C, D) using Holland classifier systems with bucket brigade strength updates and genetic algorithms. All code is self-contained — no external dependencies.
:::

Supporting documentation:

- [Quality Assessment Report](replication/2026-02-16-quality-assessment.md) — Detailed fidelity assessment against the paper
- [Comparison with DeepSeek Implementation](replication/2026-02-16-deepseek-comparison.md) — How our replication compares to an alternative implementation
- [Changelog](replication/changelog.md) — History of changes to the companion notebook

## Experiments

:::{card} Companion Notebook 2: AlphaGo Approach
:link: /jupyter/companion-notebook-2-alphago.ipynb

An alternative approach using AlphaGo-style methods (MCTS + policy-value networks) to learn trading strategies in the Kiyotaki-Wright economy.
:::

- [AlphaGo Implementations Comparison](experiments/2026-02-09-alphago-comparison.md) — Detailed comparison of two AlphaGo-style implementations
