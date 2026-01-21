"""
Experiment 001: Basic Wicksell N-tangles simulation.

This experiment runs the standard classifier system simulation
with 3 types of agents, 50 agents each, using ga3 variant.

Based on class001.m
"""

import numpy as np
from ..config import WicksellConfig
from ..classifier_simulation import ClassifierSimulation


def create_config() -> WicksellConfig:
    """Create configuration for Experiment 001."""
    
    # Generate runit vector: GA runs on even iterations with decreasing probability
    maxit = 1000
    pga = 1.0 / np.sqrt(np.arange(1, maxit // 2 + 1))
    runit = np.zeros(maxit, dtype=bool)
    # On even iterations, run GA with probability pga
    runit[1::2] = pga > np.random.rand(len(pga))
    
    config = WicksellConfig(
        maxit=maxit,
        ntypes=3,
        nagents=np.array([50, 50, 50]),
        nclasst=72,
        nclassc=12,
        bnames=np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]),
        produces=np.array([2, 3, 1]),  # Type 1 produces good 2, etc.
        storecosts=np.array([
            [0.1, 1, 20],
            [0.1, 1, 20],
            [0.1, 1, 20]
        ]),
        prodcosts=np.array([1, 1, 1]),
        utility=np.array([100, 100, 100]),
        runit=runit,
        dhist=20,
        dprob=5000,
        nback=10,
    )
    
    return config


def run(iterations: int = None, verbose: bool = True) -> ClassifierSimulation:
    """
    Run Experiment 001.
    
    Parameters
    ----------
    iterations : int, optional
        Number of iterations (default: config.maxit)
    verbose : bool
        Print progress messages
        
    Returns
    -------
    ClassifierSimulation
        The completed simulation
    """
    config = create_config()
    sim = ClassifierSimulation(config, ga_variant='ga3', verbose=verbose)
    sim.run(iterations)
    return sim


if __name__ == '__main__':
    print("Running Experiment 001: Basic Wicksell N-tangles")
    print("=" * 50)
    sim = run(iterations=100, verbose=True)
    print("\n=== Final Results ===")
    print(f"Total trades: {sim.state.total_trades}")
    print(f"Total consumptions: {sim.state.total_consumptions}")
