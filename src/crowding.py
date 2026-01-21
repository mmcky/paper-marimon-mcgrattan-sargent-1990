"""
Crowding functions for classifier system genetic algorithms.

Crowding is used to maintain diversity by replacing similar individuals.
When a new classifier (child) is created, crowding finds the most similar
weak classifier to replace, rather than replacing randomly.

Converted from crowding.m, crowding2.m, crowding3.m, crowdin3.m 
by Ellen R. McGrattan (1989).
"""

import numpy as np


def crowding_v1(child: np.ndarray, CS: np.ndarray, crowdingfactor: int,
                crowdingsubpop: int, M: int, l: int) -> int:
    """
    Find most similar weak classifier to replace (version 1).
    
    Samples random subsets of the population, finds the weakest in each subset,
    then returns the one most similar to the child.
    
    Parameters
    ----------
    child : np.ndarray
        New classifier to be inserted
    CS : np.ndarray
        Classifier system matrix (M × columns), where column l+2 (0-indexed) is strength
    crowdingfactor : int
        Number of tournaments to run
    crowdingsubpop : int
        Number of classifiers sampled per tournament
    M : int
        Total number of classifiers
    l : int
        Length of condition/action part of classifier
        
    Returns
    -------
    int
        Index of classifier to replace (0-indexed)
    """
    if crowdingfactor < 1:
        crowdingfactor = 1
    
    matchmax = -1
    mostsimilar = 0
    
    for _ in range(crowdingfactor):
        # Sample random classifiers (MATLAB: ceil(rand*M) gives 1 to M)
        worst_idx = np.random.randint(0, M, size=crowdingsubpop)
        
        # Get their strengths (column l+2 in 0-indexed is strength)
        strengths = CS[worst_idx, l + 2]
        
        # Find the weakest among sampled
        min_strength = np.min(strengths)
        weakest_mask = strengths <= min_strength
        popmember = worst_idx[np.where(weakest_mask)[0][0]]
        
        # Count matching bits between child and this classifier
        match = np.sum(child[:l] == CS[popmember, :l])
        
        if match > matchmax:
            matchmax = match
            mostsimilar = popmember
    
    return int(mostsimilar)


def crowding_v2(child: np.ndarray, CS: np.ndarray, crowdingfactor: int,
                crowdingsubpop: int, M: int, l: int) -> int:
    """
    Find most similar weak classifier to replace (version 2).
    
    Like v1, but uses column l+1 (0-indexed) for strength and includes
    action bit mismatch in similarity calculation.
    
    Parameters
    ----------
    child : np.ndarray
        New classifier to be inserted
    CS : np.ndarray
        Classifier system matrix
    crowdingfactor : int
        Number of tournaments
    crowdingsubpop : int
        Classifiers per tournament
    M : int
        Total classifiers
    l : int
        Condition length
        
    Returns
    -------
    int
        Index of classifier to replace (0-indexed)
    """
    if crowdingfactor < 1:
        crowdingfactor = 1
    
    matchmax = -1
    mostsimilar = 0
    
    for _ in range(crowdingfactor):
        worst_idx = np.random.randint(0, M, size=crowdingsubpop)
        strengths = CS[worst_idx, l + 1]  # Different column than v1
        
        min_strength = np.min(strengths)
        weakest_mask = strengths <= min_strength
        popmember = worst_idx[np.where(weakest_mask)[0][0]]
        
        # Match condition bits + bonus for different action
        match = np.sum(child[:l] == CS[popmember, :l])
        match += int(child[l] != CS[popmember, l])  # Action bit mismatch
        
        if match > matchmax:
            matchmax = match
            mostsimilar = popmember
    
    return int(mostsimilar)


def crowding_v3(child: np.ndarray, CS: np.ndarray, crowdingfactor: int,
                crowdingsubpop: float, nclass: int, lcond: int) -> int:
    """
    Find most similar weak classifier to replace (version 3).
    
    Uses nclass and lcond parameters for flexible classifier dimensions.
    
    Parameters
    ----------
    child : np.ndarray
        New classifier to be inserted
    CS : np.ndarray
        Classifier system matrix
    crowdingfactor : int
        Number of tournaments
    crowdingsubpop : float
        Proportion multiplied by nclass for sample size
    nclass : int
        Number of classifiers
    lcond : int
        Length of condition part
        
    Returns
    -------
    int
        Index of classifier to replace (0-indexed)
    """
    if crowdingfactor < 1:
        crowdingfactor = 1
    
    matchmax = -1
    mostsimilar = 0
    
    sample_size = int(crowdingsubpop * nclass)
    if sample_size < 1:
        sample_size = 1
    
    for _ in range(crowdingfactor):
        worst_idx = np.random.randint(0, nclass, size=sample_size)
        strengths = CS[worst_idx, lcond + 1]  # Strength column
        
        min_strength = np.min(strengths)
        weakest_mask = strengths <= min_strength
        popmember = worst_idx[np.where(weakest_mask)[0][0]]
        
        # Match condition + action mismatch bonus
        match = np.sum(child[:lcond] == CS[popmember, :lcond])
        match += int(child[lcond] != CS[popmember, lcond])
        
        if match > matchmax:
            matchmax = match
            mostsimilar = popmember
    
    return int(mostsimilar)


# Alias for the most commonly used version
crowding = crowding_v3
crowdin3 = crowding_v3
