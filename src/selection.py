"""
Roulette wheel selection for genetic algorithms.

Converted from select.m by Ellen R. McGrattan (1989).
"""

import numpy as np


def select(popsize: int, sumfitness: float, popf: np.ndarray) -> int:
    """
    Select an individual using roulette wheel selection.
    
    Individuals are selected with probability proportional to their fitness.
    
    Parameters
    ----------
    popsize : int
        Population size
    sumfitness : float
        Sum of all fitness values in the population
    popf : np.ndarray
        Fitness values for each individual (length popsize)
        
    Returns
    -------
    int
        Index of selected individual (0-indexed)
    """
    partsum = 0.0
    j = 0
    r = np.random.rand() * sumfitness
    
    while partsum < r and j < popsize:
        partsum += popf[j]
        j += 1
    
    # Return 0-indexed (MATLAB returns 1-indexed)
    return max(0, j - 1)


def select_pair(popsize: int, sumfitness: float, popf: np.ndarray) -> tuple:
    """
    Select two different individuals for crossover.
    
    Parameters
    ----------
    popsize : int
        Population size
    sumfitness : float
        Sum of all fitness values
    popf : np.ndarray
        Fitness values
        
    Returns
    -------
    tuple
        (mate1_idx, mate2_idx) - indices of selected individuals
    """
    mate1 = select(popsize, sumfitness, popf)
    mate2 = select(popsize, sumfitness, popf)
    
    # Ensure different individuals if population allows
    attempts = 0
    while mate2 == mate1 and attempts < 10 and popsize > 1:
        mate2 = select(popsize, sumfitness, popf)
        attempts += 1
    
    return mate1, mate2
