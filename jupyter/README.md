# Jupyter Companion Notebooks

This directory contains Jupyter notebooks that provide interactive implementations, comparisons, and extensions of the Marimon, McGrattan, and Sargent (1990) paper on artificial intelligence in economic models.

## Companion Notebooks

### 1. companion-notebook-1.ipynb

**Replication of Marimon, McGrattan, and Sargent (1990)**

A self-contained Jupyter notebook that implements and explains the Holland classifier system approach to modeling artificial intelligence in the Kiyotaki-Wright exchange economy.

**Contents:**
- Complete theoretical background on the Kiyotaki-Wright model
- Implementation of Holland's classifier systems
- Genetic algorithm for evolving trading strategies
- Bucket brigade learning mechanism
- Replication of all 8 economies from the paper (A1.1, A1.2, A2.1, A2.2, B.1, B.2, C, D)
- Economy D: Five goods, five agent types — the paper's "triumph" where the GA discovers a Nash equilibrium the authors hadn't been able to guess
- Visualization of results and equilibrium emergence

**Key Features:**
- All code is embedded in the notebook (no external dependencies on `src/`)
- Detailed mathematical exposition and economic intuition
- Reproduces Figures 1-11 from the original paper
- Demonstrates fundamental vs. speculative equilibrium selection
- Shows emergence of fiat money as medium of exchange
- Covers the five-good economy (Economy D) with 250 agents and 3-bit encoding

**Run Time:** ~5-10 minutes for all experiments (Economy D is the most compute-intensive)

---

## Comparison Documents

### compare-nb1-MMS-deepseek(tom).md

Detailed comparison between `companion-notebook-1.ipynb` (Holland classifier system) and `tom/MMS_deepseek.ipynb` (alternative implementation), analyzing differences in approach, implementation details, and results.