"""
Genetic Algorithm implementations for classifier systems.

Converted from ga.m, ga2.m, ga3.m, ga4.m by Ellen R. McGrattan (1989).

These GA variants are designed for evolving classifier systems, not for
general function optimization. For function optimization, see sga.py.
"""

import numpy as np
from typing import Tuple, Optional

from .statistics import statistics
from .selection import select
from .crowding import crowding_v1, crowding_v2, crowding_v3


def mutate_trinary(v: np.ndarray, pmutation: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply mutation to a trinary (-1, 0, 1) encoded condition string.
    
    Parameters
    ----------
    v : np.ndarray
        Condition bits (trinary encoded)
    pmutation : float
        Probability of mutating each bit
        
    Returns
    -------
    tuple
        (mutated_v, mutation_mask)
    """
    rnd = np.random.rand(len(v)) < pmutation
    # Mutation: change value by random amount, wrap to trinary
    mutations = np.ceil(np.random.rand(len(v)) * 2).astype(int) + 1
    mutated = (1 - rnd) * v + rnd * ((v + mutations + 1) % 3 - 1)
    return mutated.astype(float), rnd


def ga_v1(CS: np.ndarray, nselect: int, pcross: float, pmutation: float,
          crowdingfactor: int, crowdingsubpop: int, M: int, l2: int,
          smultiple: float, last: np.ndarray, classifier_type: int,
          iteration: int, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genetic algorithm for classifier systems (version 1).
    
    Uses single-point crossover and crowding replacement.
    
    Parameters
    ----------
    CS : np.ndarray
        Classifier system matrix (M × columns)
    nselect : int
        Number of mating pairs
    pcross : float
        Crossover probability
    pmutation : float
        Mutation probability per bit
    crowdingfactor : int
        Number of tournaments for crowding
    crowdingsubpop : int
        Sample size per tournament
    M : int
        Number of classifiers
    l2 : int
        Length of condition part
    smultiple : float
        Strength scaling multiplier (unused in this version)
    last : np.ndarray
        Indices of recently used classifiers
    classifier_type : int
        Type of classifier (affects column positions)
    iteration : int
        Current iteration number
    verbose : bool
        Print progress
        
    Returns
    -------
    tuple
        (updated_CS, updated_last)
    """
    lchrom = l2 + 2
    strength_col = lchrom  # 0-indexed: l2+2
    
    # Compute statistics
    maxs, mins, avgs, sumstrength, _ = statistics(CS[:, strength_col], M)
    
    # Handle negative strengths by shifting
    orig_mins = mins
    if mins < 0:
        CS[:, strength_col] = CS[:, strength_col] - mins
        sumstrength = np.sum(CS[:, strength_col])
    
    if verbose:
        print("Pair    Mate1   Mate2   SiteCross   Mort1   Mort2")
        print("-" * 49)
    
    ncross = 0
    nmutation = 0
    
    for j in range(nselect):
        # Select two parents
        mate1 = select(M, sumstrength, CS[:, strength_col])
        mate2 = select(M, sumstrength, CS[:, strength_col])
        
        # Crossover
        if np.random.rand() < pcross:
            jcross = 1 + int(np.floor((l2 - 1) * np.random.rand()))
            ncross += 1
        else:
            jcross = l2
        
        # Create children with mutation
        rnd1 = np.random.rand(l2) < pmutation
        rnd2 = np.random.rand(2) < pmutation
        
        v = np.concatenate([CS[mate1, :jcross], CS[mate2, jcross:l2]])
        av = (CS[mate1, l2 + 2] + CS[mate2, l2 + 2]) * 0.5
        
        # Mutate condition bits
        v_mutated = (1 - rnd1) * v + rnd1 * ((v + np.ceil(np.random.rand(l2) * 2).astype(int) + 1) % 3 - 1)
        # Mutate action bits
        action1 = np.abs(CS[mate1, l2:l2+2] - rnd2.astype(float))
        child1 = np.concatenate([v_mutated, action1, [av]])
        
        rnd3 = np.random.rand(l2) < pmutation
        rnd4 = np.random.rand(2) < pmutation
        
        v = np.concatenate([CS[mate2, :jcross], CS[mate1, jcross:l2]])
        v_mutated = (1 - rnd3) * v + rnd3 * ((v + np.ceil(np.random.rand(l2) * 2).astype(int) + 1) % 3 - 1)
        action2 = np.abs(CS[mate2, l2:l2+2] - rnd4.astype(float))
        child2 = np.concatenate([v_mutated, action2, [av]])
        
        nmutation += int(np.sum(rnd1)) + int(np.sum(rnd2)) + int(np.sum(rnd3)) + int(np.sum(rnd4))
        
        # Crowding - find classifiers to replace
        mort1 = crowding_v1(child1, CS, crowdingfactor, crowdingsubpop, M, l2)
        
        # Update last array
        matches = np.where(last[:, classifier_type] == mort1)[0]
        if len(matches) > 0:
            last[matches, classifier_type] = 0
        
        sumstrength = sumstrength - CS[mort1, l2 + 2] + av
        CS[mort1, :] = np.concatenate([child1[:l2+3], [0, 0, iteration]])
        
        mort2 = crowding_v1(child2, CS, crowdingfactor, crowdingsubpop, M, l2)
        matches = np.where(last[:, classifier_type] == mort2)[0]
        if len(matches) > 0:
            last[matches, classifier_type] = 0
        
        sumstrength = sumstrength - CS[mort2, l2 + 2] + av
        CS[mort2, :] = np.concatenate([child2[:l2+3], [0, 0, iteration]])
        
        if verbose:
            print(f"{j+1:4d}    {mate1+1:5d}   {mate2+1:5d}   {jcross:9d}   {mort1+1:5d}   {mort2+1:5d}")
    
    if verbose:
        print()
        print("Statistics Report")
        print("-----------------")
        print()
        print(f" Average strength    = {avgs:g}")
        print(f" Maximum strength    = {maxs:g}")
        print(f" Minimum strength    = {mins:g}")
        print(f" Sum of strength     = {sumstrength:g}")
        print(f" Number of crossings = {ncross:g}")
        print(f" Number of mutations = {nmutation:g}")
        print()
    
    # Restore original scale if shifted
    if orig_mins < 0:
        CS[:, strength_col] = CS[:, strength_col] + orig_mins
    
    return CS, last


def ga_v3(CS: np.ndarray, ctype: int, nselect: int, pcross: float, 
          pmutation: float, crowdingfactor: int, crowdingsubpop: float,
          nclass: int, lcond: int, last: np.ndarray, iteration: int,
          propused: float, uratio: np.ndarray, 
          verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genetic algorithm for classifier systems (version 3).
    
    Features kill-eligibility constraints based on strength and usage ratio.
    
    Parameters
    ----------
    CS : np.ndarray
        Classifier system matrix
    ctype : int
        Classifier type (1=exchange, 2=consume)
    nselect : int
        Number of mating pairs
    pcross : float
        Crossover probability
    pmutation : float
        Mutation probability
    crowdingfactor : int
        Crowding tournament count
    crowdingsubpop : float
        Crowding sample proportion
    nclass : int
        Number of classifiers
    lcond : int
        Condition length
    last : np.ndarray
        Last used classifier indices
    iteration : int
        Current iteration
    propused : float
        Proportion of most-used classifiers to consider
    uratio : np.ndarray
        [strength_threshold, usage_threshold] for kill eligibility
    verbose : bool
        Print progress
        
    Returns
    -------
    tuple
        (updated_CS, updated_last)
    """
    actpos = lcond  # 0-indexed
    fitpos = lcond + 1
    
    if ctype == 1:
        usepos = lcond + 3
        itpos = lcond + 4
    else:
        usepos = lcond + 2
        itpos = lcond + 3
    
    # Find classifiers eligible for replacement
    max_use = np.max(CS[:, usepos])
    cankill = np.where(
        (CS[:, fitpos] < uratio[0]) | 
        (CS[:, usepos] / (max_use + 1e-10) < uratio[1])
    )[0]
    
    if len(cankill) == 0:
        return CS, last
    
    numkill = len(cankill)
    if 2 * nselect > numkill:
        nselect = (numkill + numkill % 2) // 2
    
    maxb, minb, avgb, _, _ = statistics(CS[:, fitpos], nclass)
    
    orig_minb = minb
    if minb < 0:
        CS[:, fitpos] = CS[:, fitpos] - minb
    
    ncross = 0
    nmutation = 0
    ncalled = int(propused * nclass)
    
    if verbose:
        print("Pair    Mate1   Mate2   SiteCross   Mort1   Mort2  Iteration")
        print("-" * 60)
    
    for j in range(nselect):
        # Select subset of most-used classifiers
        if ncalled < nclass:
            tem1 = CS[:, usepos] + 1
            sumuse = np.sum(tem1)
            ind = []
            
            for k in range(ncalled):
                index = select(nclass, sumuse, tem1)
                ind.append(index)
                sumuse -= tem1[index]
                tem1[index] = 0
            
            ind = np.array(ind)
        else:
            ind = np.arange(nclass)
        
        lind = len(ind)
        tem1 = CS[ind, fitpos] + 1
        tem2 = np.sum(tem1)
        
        mate1_idx = select(lind, tem2, tem1)
        mate2_idx = select(lind, tem2, tem1)
        
        mate1 = ind[mate1_idx]
        mate2 = ind[mate2_idx]
        
        # Crossover
        if np.random.rand() < pcross:
            jcross = 1 + int(np.floor((lcond - 1) * np.random.rand()))
            ncross += 1
        else:
            jcross = lcond
        
        # Create children
        rnd1 = np.random.rand(lcond) < pmutation
        rnd2 = np.random.rand() < pmutation
        
        v = np.concatenate([CS[mate1, :jcross], CS[mate2, jcross:lcond]])
        av = (CS[mate1, fitpos] + CS[mate2, fitpos]) * 0.5
        
        v_mutated = (1 - rnd1) * v + rnd1 * ((v + np.ceil(np.random.rand(lcond) * 2).astype(int) + 1) % 3 - 1)
        action1 = abs(CS[mate1, actpos] - float(rnd2))
        child1 = np.concatenate([v_mutated, [action1, av]])
        
        rnd3 = np.random.rand(lcond) < pmutation
        rnd4 = np.random.rand() < pmutation
        
        v = np.concatenate([CS[mate2, :jcross], CS[mate1, jcross:lcond]])
        v_mutated = (1 - rnd3) * v + rnd3 * ((v + np.ceil(np.random.rand(lcond) * 2).astype(int) + 1) % 3 - 1)
        action2 = abs(CS[mate2, actpos] - float(rnd4))
        child2 = np.concatenate([v_mutated, [action2, av]])
        
        nmutation += int(np.sum(rnd1)) + int(rnd2) + int(np.sum(rnd3)) + int(rnd4)
        
        # Crowding on kill-eligible classifiers
        tem1 = min(nclass * crowdingsubpop / numkill, 1.0)
        mort1 = crowding_v3(child1, CS[cankill, :], crowdingfactor, tem1, numkill, lcond)
        mort1 = cankill[mort1]
        
        cankill = cankill[cankill != mort1]
        numkill -= 1
        
        if ctype == 2:
            matches = np.where(last == mort1)[0]
            if len(matches) > 0:
                last[matches] = 0
        
        if ctype == 1:
            CS[mort1, :] = np.concatenate([child1, CS[mate1, fitpos+1:usepos+1], [iteration]])
        else:
            avu = CS[mate1, usepos]
            CS[mort1, :] = np.concatenate([child1, [avu, iteration]])
        
        # Second child
        if len(cankill) == 0:
            mort2 = -1
        else:
            tem1 = min(nclass * crowdingsubpop / numkill, 1.0)
            mort2 = crowding_v3(child2, CS[cankill, :], crowdingfactor, tem1, numkill, lcond)
            mort2 = cankill[mort2]
            
            cankill = cankill[cankill != mort2]
            numkill -= 1
            
            if ctype == 2:
                matches = np.where(last == mort2)[0]
                if len(matches) > 0:
                    last[matches] = 0
            
            if ctype == 1:
                CS[mort2, :] = np.concatenate([child2, CS[mate2, fitpos+1:usepos+1], [iteration]])
            else:
                avu = 0.5 * (avu + CS[mate2, usepos])
                CS[mort2, :] = np.concatenate([child2, [avu, iteration]])
                CS[mort1, usepos] = avu
        
        if verbose:
            print(f"{j+1:4d}    {mate1+1:5d}   {mate2+1:5d}   {jcross:9d}   {mort1+1:5d}   {mort2+1:5d}  {iteration:9d}")
    
    # Restore scale
    if orig_minb < 0:
        CS[:, fitpos] = CS[:, fitpos] + orig_minb
    
    maxa, mina, avga, _, _ = statistics(CS[:, fitpos], nclass)
    
    if verbose:
        print()
        print(f"Statistics Report at iteration {iteration}")
        print("-" * 35)
        print()
        print(f" Average strength before genetics   = {avgb:g}")
        print(f" Maximum strength before genetics   = {maxb:g}")
        print(f" Minimum strength before genetics   = {minb:g}")
        print(f" Number of crossings = {ncross:g}")
        print(f" Number of mutations = {nmutation:g}")
        print(f" Average strength after genetics   = {avga:g}")
        print(f" Maximum strength after genetics   = {maxa:g}")
        print(f" Minimum strength after genetics   = {mina:g}")
        print()
    
    return CS, last


def ga_v4(CS: np.ndarray, ctype: int, nselect: int, pcross: float,
          pmutation: float, crowdingfactor: int, crowdingsubpop: float,
          nclass: int, lcond: int, last: np.ndarray, iteration: int,
          propused: float, uratio: np.ndarray,
          verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genetic algorithm for classifier systems (version 4).
    
    Uses 2-point crossover with inside/outside selection.
    
    Parameters
    ----------
    CS : np.ndarray
        Classifier system matrix
    ctype : int
        Classifier type (1=exchange, 2=consume)
    nselect : int
        Number of mating pairs
    pcross : float
        Crossover probability (unused - always crosses)
    pmutation : float
        Mutation probability (unused in this version)
    crowdingfactor : int
        Crowding tournament count
    crowdingsubpop : float
        Crowding sample proportion
    nclass : int
        Number of classifiers
    lcond : int
        Condition length
    last : np.ndarray
        Last used classifier indices
    iteration : int
        Current iteration
    propused : float
        Proportion of most-used classifiers
    uratio : np.ndarray
        [strength_threshold, usage_threshold] for kill eligibility
    verbose : bool
        Print progress
        
    Returns
    -------
    tuple
        (updated_CS, updated_last)
    """
    actpos = lcond
    fitpos = lcond + 1
    
    if ctype == 1:
        usepos = lcond + 3
        itpos = lcond + 4
    else:
        usepos = lcond + 2
        itpos = lcond + 3
    
    # Find kill-eligible classifiers
    max_use = np.max(CS[:, usepos])
    cankill = np.where(
        (CS[:, fitpos] < uratio[0]) | 
        (CS[:, usepos] / (1 + max_use) < uratio[1])
    )[0]
    
    if len(cankill) == 0:
        return CS, last
    
    numkill = len(cankill)
    if 2 * nselect > numkill:
        nselect = (numkill + numkill % 2) // 2
    
    maxb, minb, avgb, _, _ = statistics(CS[:, fitpos], nclass)
    
    orig_minb = minb
    if minb < 0:
        CS[:, fitpos] = CS[:, fitpos] - minb
    
    ncross = 0
    nmutation = 0
    ncalled = int(propused * nclass)
    
    if verbose:
        print("Pair    Mate1   Mate2  Cross Points(2)  In   Mort1   Mort2  Iteration")
        print("-" * 69)
    
    for j in range(nselect):
        # Select subset of most-used classifiers
        if ncalled < nclass:
            tem1 = CS[:, usepos] + 1
            sumuse = np.sum(tem1)
            ind = []
            
            for k in range(ncalled):
                index = select(nclass, sumuse, tem1)
                ind.append(index)
                sumuse -= tem1[index]
                tem1[index] = 0
            
            ind = np.array(ind)
        else:
            ind = np.arange(nclass)
        
        lind = len(ind)
        tem1 = CS[ind, fitpos] + 1
        tem2 = np.sum(tem1)
        
        mate1_idx = select(lind, tem2, tem1)
        mate2_idx = select(lind, tem2, tem1)
        
        mate1 = ind[mate1_idx]
        mate2 = ind[mate2_idx]
        
        av = (CS[mate1, fitpos] + CS[mate2, fitpos]) * 0.5
        child1 = np.concatenate([CS[mate1, :lcond+1], [av]])
        child2 = np.concatenate([CS[mate2, :lcond+1], [av]])
        
        # 2-point crossover
        jcross = np.sort(1 + np.floor((lcond + 1) * np.random.rand(2)).astype(int))
        inside = np.random.rand() > 0.5
        
        if inside:
            cross_ind = np.arange(jcross[0], jcross[1])
        else:
            cross_ind = np.concatenate([np.arange(0, jcross[0]), np.arange(jcross[1], lcond)])
        
        if len(cross_ind) > 0:
            tem1_vals = child1[cross_ind].copy()
            tem2_vals = child2[cross_ind].copy()
            
            # Swap with handling for negative values
            tem3 = np.abs(
                np.where(tem1_vals < 0, tem2_vals, tem1_vals) - 
                np.where(tem2_vals < 0, tem1_vals, tem2_vals)
            )
            
            child1[cross_ind] = tem1_vals * (~tem3.astype(bool)).astype(float) - tem3
            child2[cross_ind] = tem2_vals * (~tem3.astype(bool)).astype(float) - tem3
        
        # Count as crossover if actually crossed
        if not (inside and jcross[0] != jcross[1]) or (not inside and jcross[1] - jcross[0] == lcond):
            ncross += 1
        
        # Crowding
        tem1 = min(nclass * crowdingsubpop / numkill, 1.0)
        mort1 = crowding_v3(child1, CS[cankill, :], crowdingfactor, tem1, numkill, lcond)
        mort1 = cankill[mort1]
        
        cankill = cankill[cankill != mort1]
        numkill -= 1
        
        if ctype == 2:
            matches = np.where(last == mort1)[0]
            if len(matches) > 0:
                last[matches] = 0
        
        if ctype == 1:
            CS[mort1, :] = np.concatenate([child1, CS[mate1, fitpos+1:usepos+1], [iteration]])
        else:
            avu = CS[mate1, usepos]
            CS[mort1, :] = np.concatenate([child1, [avu, iteration]])
        
        # Second child
        if len(cankill) == 0:
            mort2 = -1
        else:
            tem1 = min(nclass * crowdingsubpop / numkill, 1.0)
            mort2 = crowding_v3(child2, CS[cankill, :], crowdingfactor, tem1, numkill, lcond)
            mort2 = cankill[mort2]
            
            cankill = cankill[cankill != mort2]
            numkill -= 1
            
            if ctype == 2:
                matches = np.where(last == mort2)[0]
                if len(matches) > 0:
                    last[matches] = 0
            
            if ctype == 1:
                CS[mort2, :] = np.concatenate([child2, CS[mate2, fitpos+1:usepos+1], [iteration]])
            else:
                avu = 0.5 * (avu + CS[mate2, usepos])
                CS[mort2, :] = np.concatenate([child2, [avu, iteration]])
                CS[mort1, usepos] = avu
        
        if verbose:
            print(f"{j+1:4d}    {mate1+1:5d}   {mate2+1:5d}  {jcross[0]:7d} {jcross[1]:7d}   {int(inside):2d}   {mort1+1:5d}   {mort2+1:5d}  {iteration:9d}")
    
    # Restore scale
    if orig_minb < 0:
        CS[:, fitpos] = CS[:, fitpos] + orig_minb
    
    maxa, mina, avga, _, _ = statistics(CS[:, fitpos], nclass)
    
    if verbose:
        print()
        print(f"Statistics Report at iteration {iteration}")
        print("-" * 35)
        print()
        print(f" Average strength before genetics   = {avgb:g}")
        print(f" Maximum strength before genetics   = {maxb:g}")
        print(f" Minimum strength before genetics   = {minb:g}")
        print(f" Number of crossings = {ncross:g}")
        print(f" Number of mutations = {nmutation:g}")
        print(f" Average strength after genetics   = {avga:g}")
        print(f" Maximum strength after genetics   = {maxa:g}")
        print(f" Minimum strength after genetics   = {mina:g}")
        print()
    
    return CS, last


# Aliases for compatibility
ga = ga_v1
ga2 = ga_v1  # Similar to v1 with minor differences
ga3 = ga_v3
ga4 = ga_v4
