"""
Fitness scaling functions for genetic algorithms.

Converted from scalepop.m and scalestr.m by Ellen R. McGrattan (1989).
"""

import numpy as np
from typing import Tuple


def scale_population(popo: np.ndarray, maxf: float, minf: float, 
                     avgf: float, fmultiple: float = 2.0) -> Tuple[np.ndarray, float]:
    """
    Linear scaling of population fitness values.
    
    Scales fitness values to handle negative fitness and control selection pressure.
    The scaling ensures that the average fitness scales to fmultiple times itself,
    while keeping fitness values non-negative.
    
    Parameters
    ----------
    popo : np.ndarray
        Original (unscaled) fitness values
    maxf : float
        Maximum fitness in population
    minf : float
        Minimum fitness in population
    avgf : float
        Average fitness in population
    fmultiple : float, optional
        Target ratio of max scaled fitness to average (default 2.0)
        
    Returns
    -------
    tuple
        (popf, sumfitness)
        - popf: scaled fitness values
        - sumfitness: sum of scaled fitness values
    """
    newmin = 1.0
    
    if avgf > 0:
        # Check which scaling formula to use
        threshold = (fmultiple * avgf - maxf + (maxf - avgf) * newmin / avgf) / (fmultiple - 1)
        
        if minf > threshold:
            # Scale to get fmultiple * avg as max
            delta = maxf - avgf
            a = (fmultiple - 1) * avgf / delta
            b = avgf * (maxf - fmultiple * avgf) / delta
        else:
            # Scale to get newmin as minimum
            delta = avgf - minf
            a = (avgf - newmin) / delta
            b = newmin - a * minf
    else:
        # Handle non-positive average
        a = 1.0
        b = abs(minf) + newmin
    
    # Apply scaling with floor at 1
    popf = np.maximum(a * popo + b, np.ones(len(popo)))
    sumfitness = float(np.sum(popf))
    
    return popf, sumfitness


def scale_strength(maxs: float, mins: float, avgs: float, 
                   smultiple: float = 2.0) -> Tuple[float, float]:
    """
    Compute linear scaling coefficients for classifier strength.
    
    Returns coefficients (a, b) such that scaled_strength = a * strength + b.
    
    Parameters
    ----------
    maxs : float
        Maximum strength in population
    mins : float
        Minimum strength in population
    avgs : float
        Average strength in population
    smultiple : float, optional
        Target ratio of max scaled strength to average (default 2.0)
        
    Returns
    -------
    tuple
        (a, b) - linear scaling coefficients
    """
    if avgs > 0:
        threshold = (smultiple * avgs - maxs) / (smultiple - 1)
        
        if mins > threshold:
            # Scale to get smultiple * avg as max
            delta = maxs - avgs
            a = (smultiple - 1) * avgs / delta
            b = avgs * (maxs - smultiple * avgs) / delta
        else:
            # Scale to ensure non-negative
            delta = avgs - mins
            a = avgs / delta
            b = -mins * avgs / delta
    else:
        # Handle non-positive average
        a = 1.0
        b = abs(mins)
    
    return a, b


# Aliases for compatibility with original MATLAB naming
scalepop = scale_population
scalestr = scale_strength
