# Python Replication of Original MATLAB Code

> **Status: Work in Progress**

This directory contains a Python port of the original MATLAB code by Ellen R. McGrattan (1989) that implements the genetic algorithms and classifier systems described in:

> Marimon, R., McGrattan, E., & Sargent, T. J. (1990). Money as a Medium of Exchange in an Economy with Artificially Intelligent Agents. *Journal of Economic Dynamics and Control*, 14(2), 329–373.

The original MATLAB source files are preserved in [`original/matlab/`](../original/matlab/).

## Modules

| Module | MATLAB Origin | Description |
|--------|--------------|-------------|
| `config.py` | — | Configuration dataclasses for SGA and Wicksell simulations |
| `decode.py` | `decode.m` | Binary string to real number conversion |
| `selection.py` | `select.m` | Roulette wheel selection |
| `statistics.py` | `statistics.m` | Population statistics (max, min, avg, sum fitness) |
| `objfunc.py` | `objfunc.m` | Objective function (Rosenbrock) |
| `scaling.py` | `scalepop.m`, `scalestr.m` | Linear fitness scaling and classifier strength scaling |
| `crowding.py` | `crowding.m` | Crowding for population diversity |
| `create.py` | `create.m` | Classifier creation |
| `ga.py` | `ga.m`–`ga4.m` | Genetic algorithm variants |
| `sga.py` | `sga.m` | Simple Genetic Algorithm |
| `classifier_simulation.py` | `wicksell.m` | Main classifier system simulation |
| `visualization.py` | — | Plotting utilities (matplotlib) |

## Experiments

The `experiments/` subdirectory contains simulation scripts:

- `experiment_001.py` — Basic Wicksell simulation
- `experiment_002.py` — Extended statistics tracking
- `experiment_003.py` — Participation tracking
- `experiment_004.py` — GA4 (2-point crossover)

## Usage

```bash
pip install -r requirements.txt
```

```python
from src.sga import simple_genetic_algorithm
from src.config import SGAConfig

config = SGAConfig()
result = simple_genetic_algorithm(config)
```

## Relationship to Companion Notebook

The self-contained [companion notebook](../website/replication/companion-notebook-1.ipynb) is the primary replication effort and does not depend on this `src/` code. This module-based implementation is a separate, parallel effort to create reusable Python components from the original MATLAB.
