"""
Population statistics for genetic algorithms.

Converted from statistics.m by Ellen R. McGrattan (1989).
"""

import numpy as np
from typing import Tuple, Optional


def statistics(popf: np.ndarray, popsize: Optional[int] = None) -> Tuple[float, float, float, float, int]:
    """
    Compute population fitness statistics.
    
    Parameters
    ----------
    popf : np.ndarray
        Fitness values for each individual
    popsize : int, optional
        Population size. If None, inferred from popf length.
        
    Returns
    -------
    tuple
        (maxf, minf, avg, sumfitness, best_idx)
        - maxf: maximum fitness
        - minf: minimum fitness  
        - avg: average fitness
        - sumfitness: sum of all fitness values
        - best_idx: index of best individual (0-indexed)
    """
    if popsize is None:
        popsize = len(popf)
    
    maxf = float(np.max(popf))
    minf = float(np.min(popf))
    sumfitness = float(np.sum(popf))
    avg = sumfitness / popsize
    
    # Find index of best individual (handle ties by taking first)
    best_idx = int(np.argmax(popf))
    
    return maxf, minf, avg, sumfitness, best_idx
