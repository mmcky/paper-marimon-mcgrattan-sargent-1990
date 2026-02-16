---
title: "Money as a Medium of Exchange in an Economy with Artificially Intelligent Agents"
---

# Replication of Marimon, McGrattan & Sargent (1990)

This project provides a Python replication of the classifier system simulations from:

> Marimon, R., McGrattan, E., & Sargent, T. J. (1990). **Money as a Medium of Exchange in an Economy with Artificially Intelligent Agents.** *Journal of Economic Dynamics and Control*, 14(2), 329–373. [DOI: 10.1016/0165-1889(90)90025-C](https://doi.org/10.1016/0165-1889(90)90025-C)

The paper studies how artificially intelligent agents using Holland's classifier systems learn to use commodity money in a Kiyotaki-Wright economy. Agents are endowed with production technologies and must trade to obtain their desired consumption goods. Through a bucket brigade payment system and genetic algorithms, agents learn which goods to accept in trade — and commodity money emerges endogenously.

## Companion Notebooks

The core replication is contained in two Jupyter notebooks:

::::{grid} 1 1 2 2
:gutter: 3

:::{card} Companion Notebook 1
:link: /jupyter/companion-notebook-1.ipynb

Full replication of all **8 economies** from the paper (A1.1, A1.2, A2.1, A2.2, B.1, B.2, C, D) using Holland classifier systems with bucket brigade strength updates and genetic algorithms.
:::

:::{card} Companion Notebook 2
:link: /jupyter/companion-notebook-2-alphago.ipynb

An alternative approach using AlphaGo-style methods to learn trading strategies in the Kiyotaki-Wright economy.
:::

::::

## The Paper

The paper explores whether artificially intelligent agents can learn to adopt a commodity as money — a medium of exchange — in a decentralized trading environment. The key ingredients are:

- **Kiyotaki-Wright (1989) economy**: Multiple agent types with different production technologies, indivisible goods, and random pairwise matching
- **Holland classifier systems**: Trinary-encoded rules (0, 1, #) that compete via an auction mechanism to make trade and consumption decisions
- **Bucket brigade**: A payment system (Equations 10–11) that propagates rewards backward through the chain of classifiers responsible for successful outcomes
- **Genetic algorithm**: Periodic evolution of classifier populations through crossover and mutation to discover better trading strategies

The paper simulates 8 economies with varying parameters and demonstrates convergence to Nash-Markov equilibria, including the emergence of fiat money in Economy C.

## Additional Documentation

- [Algorithm Description](algorithm.md) — Details of the genetic algorithm implementation
- [Code Overview](readme.md) — Python source code structure and usage
- [Paper Text](../paper/paper.md) — Full extracted text of the original paper
