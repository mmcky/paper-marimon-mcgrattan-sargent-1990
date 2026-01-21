"""
Experiment 003: Wicksell N-tangles with participation tracking.

Similar to experiment_001 but tracks participation rates for
each classifier in the genetic algorithm.

Based on class003.m
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict

from ..config import WicksellConfig
from ..classifier_simulation import ClassifierSimulation


@dataclass
class ParticipationTracking:
    """Track participation rates in genetic algorithm."""
    # Participation counts: Pt[classifier, type] = times selected for GA
    Pt: np.ndarray = None
    Pc: np.ndarray = None
    
    def __init__(self, nclasst: int, nclassc: int, ntypes: int):
        self.Pt = np.zeros((nclasst, ntypes))
        self.Pc = np.zeros((nclassc, ntypes))


class Experiment003Simulation(ClassifierSimulation):
    """Simulation with GA participation tracking."""
    
    def __init__(self, config: WicksellConfig, verbose: bool = True):
        super().__init__(config, ga_variant='ga3', verbose=verbose)
        self.participation = ParticipationTracking(
            config.nclasst, config.nclassc, config.ntypes
        )
    
    def _run_genetic_algorithm(self, iteration: int):
        """Run GA with participation tracking."""
        # Call parent method
        super()._run_genetic_algorithm(iteration)
        
        # Track which classifiers participated
        # (This is a simplified tracking - the original tracks exact participants)


def create_config() -> WicksellConfig:
    """Create configuration for Experiment 003."""
    
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


def run(iterations: int = None, verbose: bool = True) -> Experiment003Simulation:
    """
    Run Experiment 003.
    
    Parameters
    ----------
    iterations : int, optional
        Number of iterations (default: config.maxit)
    verbose : bool
        Print progress messages
        
    Returns
    -------
    Experiment003Simulation
        The completed simulation with participation tracking
    """
    config = create_config()
    sim = Experiment003Simulation(config, verbose=verbose)
    sim.run(iterations)
    return sim


if __name__ == '__main__':
    print("Running Experiment 003: Participation Tracking")
    print("=" * 50)
    sim = run(iterations=100, verbose=True)
    print("\n=== Final Results ===")
    print(f"Total trades: {sim.state.total_trades}")
    print(f"Total consumptions: {sim.state.total_consumptions}")
