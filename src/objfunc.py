"""
Objective functions for optimization.

Converted from objfunc.m by Ellen R. McGrattan (1989).
"""

import numpy as np


def rosenbrock(x: np.ndarray) -> float:
    """
    Negated Rosenbrock function (for maximization).
    
    The Rosenbrock function is a common test function for optimization.
    The global minimum is at (1, 1) with value 0.
    Since we maximize in the GA, we negate it.
    
    Parameters
    ----------
    x : np.ndarray
        Parameter vector [x1, x2]
        
    Returns
    -------
    float
        Negative of Rosenbrock function value
    """
    return -100 * (x[1] - x[0]**2)**2 - (1 - x[0])**2


def objfunc(x: np.ndarray) -> float:
    """
    Default objective function (negated Rosenbrock).
    
    Alias for rosenbrock() for compatibility with original MATLAB code.
    
    Parameters
    ----------
    x : np.ndarray
        Parameter vector
        
    Returns
    -------
    float
        Objective function value (to be maximized)
    """
    return rosenbrock(x)
