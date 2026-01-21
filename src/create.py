"""
Functions for creating new classifiers when no match exists.

Converted from create.m and cre.m by Ellen R. McGrattan (1989).
"""

import numpy as np
from typing import Tuple


def create(CS: np.ndarray, M: int, l2: int, new: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Create a new classifier by replacing the most redundant or weakest.
    
    When a condition has no matching classifier, this function finds
    the best classifier to replace:
    1. First tries to find redundant classifiers (duplicates)
    2. If none redundant, replaces the weakest
    
    Parameters
    ----------
    CS : np.ndarray
        Classifier system matrix (M × columns)
    M : int
        Number of classifiers
    l2 : int
        Length of condition part
    new : np.ndarray
        New condition to add (length l2)
        
    Returns
    -------
    tuple
        (replaced_index, updated_CS)
    """
    row, col = CS.shape
    x = CS[:, :l2].copy()
    lx = M
    
    most = 1
    info = x[0, :].copy()
    
    # Find the most redundant condition pattern
    while lx >= 1:
        # Find indices where condition differs from first row
        diffs = np.sum(np.abs(x - x[0:1, :]), axis=1)
        ind = np.where(diffs > 0)[0]
        
        li = lx - len(ind)
        if li > most:
            most = li
            info = x[0, :].copy()
        
        if len(ind) == 0:
            break
            
        x = x[ind, :]
        lx = len(ind)
    
    # Determine which classifier to replace
    strength_col = l2 + 1  # 0-indexed
    
    if li == 1:
        # No redundancy - replace weakest
        min_strength = np.min(CS[:, strength_col])
        win_candidates = np.where(CS[:, strength_col] <= min_strength)[0]
        lw = len(win_candidates)
        if lw > 1:
            win = win_candidates[int(np.ceil(np.random.rand() * lw)) - 1]
        else:
            win = win_candidates[0]
    else:
        # Found redundant classifiers - replace weakest among them
        matches = np.sum(np.abs(CS[:, :l2] - info), axis=1)
        ind = np.where(matches == 0)[0]
        
        c_strengths = CS[ind, strength_col]
        min_strength = np.min(c_strengths)
        win_candidates = np.where(c_strengths <= min_strength)[0]
        lw = len(win_candidates)
        
        win = ind[win_candidates[int(np.ceil(np.random.rand() * lw)) - 1]]
    
    # Create new classifier
    # [condition, random_action, average_strength, zeros for remaining columns]
    avg_strength = np.mean(CS[:, strength_col])
    new_classifier = np.zeros(col)
    new_classifier[:l2] = new
    new_classifier[l2] = round(np.random.rand())  # Random action
    new_classifier[l2 + 1] = avg_strength
    # Remaining columns stay zero
    
    CS[win, :] = new_classifier
    
    return int(win), CS


def create_classifier(CS: np.ndarray, nclass: int, lcond: int, 
                      condition: np.ndarray, iteration: int = 0) -> Tuple[int, np.ndarray]:
    """
    Create a new classifier for an unmatched condition (alternate version).
    
    Parameters
    ----------
    CS : np.ndarray
        Classifier system matrix
    nclass : int
        Number of classifiers
    lcond : int
        Condition length
    condition : np.ndarray
        New condition to add
    iteration : int
        Current iteration (stored in classifier)
        
    Returns
    -------
    tuple
        (replaced_index, updated_CS)
    """
    return create(CS, nclass, lcond, condition)


# Alias for compatibility with original MATLAB naming
cre = create
