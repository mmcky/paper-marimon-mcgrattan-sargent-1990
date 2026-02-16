# Genetic Algorithms and Classifier Systems

This project contains Python implementations of genetic algorithms and classifier 
systems for studying the emergence of commodity money in a Wicksell-type economy.

Originally implemented in MATLAB by Ellen R. McGrattan (1989), this code has been
converted to Python with modern idioms and NumPy vectorization.

## Project Structure

```
src/
├── __init__.py
├── config.py           # Configuration dataclasses
├── decode.py           # Binary to real number conversion
├── select.py           # Roulette wheel selection
├── statistics.py       # Population statistics
├── objfunc.py          # Objective functions (Rosenbrock)
├── scaling.py          # Fitness scaling
├── crowding.py         # Crowding for diversity
├── ga.py               # Genetic algorithm variants
├── create.py           # Classifier creation
├── sga.py              # Simple Genetic Algorithm
├── classifier_simulation.py  # Main classifier simulation
└── experiments/
    ├── __init__.py
    ├── experiment_001.py  # Basic Wicksell simulation
    ├── experiment_002.py  # Extended statistics
    ├── experiment_003.py  # Participation tracking
    └── experiment_004.py  # GA4 (2-point crossover)

docs/
├── index.md            # Documentation home
├── readme.md           # This file
└── algorithm.md        # Algorithm description
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running the Simple Genetic Algorithm

```python
from src.sga import simple_genetic_algorithm
from src.config import SGAConfig

config = SGAConfig(
    popsize=50,
    maxgen=100,
    pcross=0.6,
    pmutation=0.01
)

result = simple_genetic_algorithm(config)
print(f"Best solution: {result['best_params']}")
print(f"Best fitness: {result['best_value']}")
```

### Running Wicksell Simulations

```python
from src.experiments import experiment_001

# Run with default parameters
sim = experiment_001.run(iterations=100, verbose=True)

# Access results
print(f"Total trades: {sim.state.total_trades}")
print(f"Total consumptions: {sim.state.total_consumptions}")
```

## Algorithms

### Simple Genetic Algorithm (SGA)

The SGA optimizes real-valued functions using binary encoding:

- **Encoding**: Real parameters encoded as binary strings
- **Selection**: Roulette wheel based on fitness
- **Crossover**: Single-point crossover with probability `pcross`
- **Mutation**: Bit-flip mutation with probability `pmutation`
- **Scaling**: Fitness scaling to maintain selection pressure

### Classifier Systems

Classifier systems are rule-based learning systems:

- **Classifiers**: Condition → Action rules with strengths
- **Matching**: Find rules matching current state (supports wildcards)
- **Reward**: Successful rules increase in strength
- **Evolution**: GA evolves better classifiers over time

### Wicksell Economy

The simulation models a commodity money economy:

- **Agents**: Different types with different production specializations
- **Trade**: Agents randomly meet and decide whether to trade
- **Consumption**: Agents consume goods they hold
- **Learning**: Classifier systems learn trading strategies

## References

- McGrattan, E. R. (1989). "MATLAB Code for Genetic Algorithms and Classifier Systems"
- Holland, J. H. (1975). "Adaptation in Natural and Artificial Systems"
- Wicksell, K. (1898). "Interest and Prices"
