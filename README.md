# Money as a Medium of Exchange in an Economy with Artificially Intelligent Agents

**Marimon, McGrattan, and Sargent (1990)**

> ⚠️ **Experimental**: This is an experimental port of the original MATLAB code to Python. The code is provided as-is for research and educational purposes.

## About

This repository contains a Python implementation of the classifier systems and genetic algorithms used in the paper:

> Marimon, R., McGrattan, E., & Sargent, T. J. (1990). Money as a Medium of Exchange in an Economy with Artificially Intelligent Agents. *Journal of Economic Dynamics and Control*, 14(2), 329-373.

The paper studies how artificially intelligent agents using Holland's classifier systems learn to use commodity money in a Wicksell-type economy, reproducing and extending the theoretical model of Kiyotaki and Wright (1989).

## Project Structure

```
├── src/                    # Python implementation
│   ├── config.py           # Configuration dataclasses
│   ├── classifier_simulation.py  # Main simulation
│   ├── ga.py               # Genetic algorithm variants
│   ├── sga.py              # Simple Genetic Algorithm
│   └── ...
├── paper/                  # MyST paper source
│   ├── paper.md            # Main paper in MyST Markdown
│   └── figures/            # Generated figures
├── scripts/                # Utility scripts
│   └── generate_figures.py # Figure generation
├── tests/                  # Test suite
├── docs/                   # Documentation
└── data/mms/               # Original MATLAB code
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running Simulations

```python
from src.experiments import experiment_001

# Run Wicksell economy simulation
sim = experiment_001.run(iterations=1000, verbose=True)

# Access results
print(f"Total trades: {sim.state.total_trades}")
print(f"Total consumptions: {sim.state.total_consumptions}")
```

## Generating Figures

```bash
python scripts/generate_figures.py --economy A --output-dir paper/figures
```

## Building the Paper

The paper is written in MyST Markdown and can be built with:

```bash
# Install mystmd
npm install -g mystmd

# Build HTML
npx myst build --html

# Or start development server
npx myst start
```

## Running Tests

```bash
pytest tests/ -v
```

## Original Code

The original MATLAB code by Ellen R. McGrattan (1989) is preserved in `data/mms/` for reference.

## License

This is an academic reproduction for research and educational purposes.

## References

- Kiyotaki, N., & Wright, R. (1989). On Money as a Medium of Exchange. *Journal of Political Economy*, 97(4), 927-954.
- Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
- Holland, J. H. (1986). Escaping Brittleness: The Possibilities of General-Purpose Learning Algorithms Applied to Parallel Rule-Based Systems. In *Machine Learning: An Artificial Intelligence Approach* (Vol. 2).
