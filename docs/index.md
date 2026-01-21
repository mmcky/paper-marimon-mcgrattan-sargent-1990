# Genetic Algorithms and Classifier Systems

A Python implementation of genetic algorithms and classifier systems for
studying the emergence of commodity money in economic simulations.

## Overview

This project implements genetic algorithms (GAs) and classifier systems
originally developed by Ellen R. McGrattan (1989) in MATLAB. The code has
been modernized with:

- Python 3.10+ compatibility
- NumPy vectorization for performance
- Dataclass-based configuration
- Type hints and documentation
- Modular, testable design

## Contents

See the navigation for full documentation:

- [Code Overview](readme.md) - Original MATLAB code documentation
- [Algorithm Description](algorithm.md) - Genetic algorithm details

## Key Components

### Simple Genetic Algorithm (SGA)

The `sga.py` module implements a standard genetic algorithm for function
optimization. It supports:

- Binary encoding of real-valued parameters
- Roulette wheel selection
- Single-point crossover
- Bit-flip mutation
- Fitness scaling
- Gradient checking for local search

### Classifier Systems

Classifier systems are rule-based learning systems where rules (classifiers)
compete to provide actions. The implementation includes:

- Trade classifiers: decide whether to trade goods
- Consume classifiers: decide whether to consume goods
- Wildcard matching for general rules
- Strength-based competition

### Wicksell Economy Simulation

The main simulation models a Wicksell-type economy where:

- Multiple agent types produce different goods
- Agents meet randomly and can trade
- Classifier systems learn trading strategies
- Commodity money can emerge endogenously

## Quick Start

```python
# Run a simple experiment
from src.experiments import experiment_001

sim = experiment_001.run(iterations=100)
print(f"Trades: {sim.state.total_trades}")
```

## References

- Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*
- McGrattan, E. R. (1989). MATLAB Code for Genetic Algorithms
- Wicksell, K. (1898). *Interest and Prices*
