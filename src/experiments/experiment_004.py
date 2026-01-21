"""
Experiment 004: Wicksell N-tangles with GA4 (2-point crossover).

Uses the ga4 variant which implements 2-point crossover instead
of single-point crossover.

Based on class004.m
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List

from ..config import WicksellConfig
from ..classifier_simulation import ClassifierSimulation


class Experiment004Simulation(ClassifierSimulation):
    """Simulation using GA4 (2-point crossover) variant."""
    
    def __init__(self, config: WicksellConfig, verbose: bool = True):
        super().__init__(config, ga_variant='ga4', verbose=verbose)


def create_config() -> WicksellConfig:
    """Create configuration for Experiment 004."""
    
    maxit = 1000
    pga = 1.0 / np.sqrt(np.arange(1, maxit // 2 + 1))
    runit = np.zeros(maxit, dtype=bool)
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
        produces=np.array([2, 3, 1]),
        storecosts=np.array([
            [0.1, 1, 20],
            [0.1, 1, 20],
            [0.1, 1, 20]
        ]),
        prodcosts=np.array([1, 1, 1]),
        utility=np.array([100, 100, 100]),
        runit=runit,
        dhist=20,
    )
    
    return config


def run(iterations: int = None, verbose: bool = True) -> Experiment004Simulation:
    """
    Run Experiment 004.
    
    Parameters
    ----------
    iterations : int, optional
        Number of iterations (default: config.maxit)
    verbose : bool
        Print progress messages
        
    Returns
    -------
    Experiment004Simulation
        The completed simulation using GA4 variant
    """
    config = create_config()
    sim = Experiment004Simulation(config, verbose=verbose)
    sim.run(iterations)
    return sim


if __name__ == '__main__':
    print("Running Experiment 004: GA4 (2-point crossover)")
    print("=" * 50)
    sim = run(iterations=100, verbose=True)
    print("\n=== Final Results ===")
    print(f"Total trades: {sim.state.total_trades}")
    print(f"Total consumptions: {sim.state.total_consumptions}")
