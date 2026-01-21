"""
Decode binary chromosome strings to real parameter values.

Converted from decode.m by Ellen R. McGrattan (1989).
"""

import numpy as np


def decode(popc: np.ndarray, nparms: int, lparm: np.ndarray, 
           maxparm: np.ndarray, minparm: np.ndarray) -> np.ndarray:
    """
    Convert binary chromosome strings to real parameter values.
    
    Parameters
    ----------
    popc : np.ndarray
        Binary population matrix (popsize × lchrom), where each row is a chromosome
    nparms : int
        Number of parameters encoded in each chromosome
    lparm : np.ndarray
        Array of length nparms, specifying bits per parameter
    maxparm : np.ndarray
        Maximum values for each parameter
    minparm : np.ndarray
        Minimum values for each parameter
        
    Returns
    -------
    np.ndarray
        Decoded parameter matrix (popsize × nparms)
    """
    popsize = popc.shape[0]
    x = np.zeros((popsize, nparms))
    
    for j in range(nparms):
        # Starting position in chromosome (0-indexed)
        s = int(np.sum(lparm[:j]))
        l = int(lparm[j])
        
        # Extract bits for this parameter
        bits = popc[:, s:s+l]
        
        # Binary to decimal conversion: sum(bit_i * 2^i)
        powers = 2 ** np.arange(l)
        decimal_val = bits @ powers
        
        # Scale to parameter range
        x[:, j] = minparm[j] + (maxparm[j] - minparm[j]) * decimal_val / (2**l - 1)
    
    return x
