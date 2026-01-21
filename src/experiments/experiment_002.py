"""
Experiment 002: Wicksell N-tangles with extended statistics storage.

Similar to experiment_001 but with additional frequency storage
for historical analysis.

Based on class002.m
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

from ..config import WicksellConfig
from ..classifier_simulation import ClassifierSimulation, SimulationState, AgentType


@dataclass  
class ExtendedStatistics:
    """Extended statistics storage for historical tracking."""
    shist: List[np.ndarray] = field(default_factory=list)
    sfreq1: Dict[int, List[np.ndarray]] = field(default_factory=dict)
    sfreq2: Dict[int, List[np.ndarray]] = field(default_factory=dict)
    sfreq3: Dict[int, List[np.ndarray]] = field(default_factory=dict)
    sfreq4: Dict[int, List[np.ndarray]] = field(default_factory=dict)
    sfreq5: Dict[int, List[np.ndarray]] = field(default_factory=dict)


class Experiment002Simulation(ClassifierSimulation):
    """Extended simulation with historical statistics storage."""
    
    def __init__(self, config: WicksellConfig, verbose: bool = True):
        super().__init__(config, ga_variant='ga3', verbose=verbose)
        self.extended_stats = ExtendedStatistics()
        
        # Initialize storage for each type
        for i in range(config.ntypes):
            self.extended_stats.sfreq1[i] = []
            self.extended_stats.sfreq2[i] = []
            self.extended_stats.sfreq3[i] = []
            self.extended_stats.sfreq4[i] = []
            self.extended_stats.sfreq5[i] = []
    
    def _save_frequencies(self, iteration: int):
        """Save frequency matrices at specified intervals."""
        cfg = self.config
        
        if iteration % cfg.savef == 0:
            for type_id, agent in self.state.agent_types.items():
                self.extended_stats.sfreq1[type_id].append(agent.freq1.copy())
                self.extended_stats.sfreq2[type_id].append(agent.freq2.copy())
                self.extended_stats.sfreq3[type_id].append(agent.freq3.copy())
                self.extended_stats.sfreq4[type_id].append(agent.freq4.copy())
                self.extended_stats.sfreq5[type_id].append(agent.freq5.copy())


def create_config() -> WicksellConfig:
    """Create configuration for Experiment 002."""
    
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
        savef=250,  # Save frequencies every 250 iterations
    )
    
    return config


def run(iterations: int = None, verbose: bool = True) -> Experiment002Simulation:
    """
    Run Experiment 002.
    
    Parameters
    ----------
    iterations : int, optional
        Number of iterations (default: config.maxit)
    verbose : bool
        Print progress messages
        
    Returns
    -------
    Experiment002Simulation
        The completed simulation with extended statistics
    """
    config = create_config()
    sim = Experiment002Simulation(config, verbose=verbose)
    sim.run(iterations)
    return sim


if __name__ == '__main__':
    print("Running Experiment 002: Extended Statistics")
    print("=" * 50)
    sim = run(iterations=100, verbose=True)
    print("\n=== Final Results ===")
    print(f"Total trades: {sim.state.total_trades}")
    print(f"Total consumptions: {sim.state.total_consumptions}")
