"""
Simple Genetic Algorithm (SGA) for function optimization.

A genetic algorithm for finding the maximum of a function subject to 
simple bounds on the parameters.

Converted from sga.m by Ellen R. McGrattan (1989).

References:
-----------
[1] Goldberg, David, GENETIC ALGORITHMS FOR OPTIMIZATION, SEARCH, AND
    MACHINE LEARNING, (Menlo Park, CA: Addison-Wesley Co.), 1989.
"""

import numpy as np
from typing import Callable, Tuple, Optional
from dataclasses import dataclass
import pickle

from .decode import decode
from .statistics import statistics
from .scaling import scale_population
from .selection import select


@dataclass
class SGAResult:
    """Results from Simple Genetic Algorithm optimization."""
    xbest: np.ndarray       # Best parameter vector found
    fbest: float            # Best objective function value
    gbest: np.ndarray       # Gradient at best point (if computed)
    code: int               # Exit code: 1=optimal, 2=max iterations
    generations: int        # Number of generations run
    
    def __str__(self) -> str:
        return (f"SGAResult(fbest={self.fbest:.6e}, "
                f"generations={self.generations}, code={self.code})")


def numerical_gradient(func: Callable, x: np.ndarray, 
                       typsiz: np.ndarray, side: np.ndarray) -> np.ndarray:
    """
    Compute numerical gradient using finite differences.
    
    Parameters
    ----------
    func : Callable
        Objective function
    x : np.ndarray
        Point at which to compute gradient
    typsiz : np.ndarray
        Typical size of parameters (for step size)
    side : np.ndarray
        1 for one-sided, 2 for two-sided differences
        
    Returns
    -------
    np.ndarray
        Gradient vector
    """
    nparam = len(x)
    g = np.zeros(nparam)
    
    for i in range(nparam):
        h = np.sqrt(1e-15) * max(abs(x[i]), typsiz[i])
        
        x_plus = x.copy()
        x_plus[i] += h
        
        if side[i] == 2:
            x_minus = x.copy()
            x_minus[i] -= h
            g[i] = (func(x_plus) - func(x_minus)) / (2 * h)
        else:
            g[i] = (func(x_plus) - func(x)) / h
    
    return g


def sga(func: Callable[[np.ndarray], float],
        lb: np.ndarray,
        ub: np.ndarray,
        popsize: int = 50,
        maxgen: int = 50,
        lparam: Optional[np.ndarray] = None,
        pcrossover: float = 0.6,
        pmutation: float = 0.01,
        gradchk: bool = True,
        gtol: float = 1e-5,
        agflag: bool = False,
        side: Optional[np.ndarray] = None,
        typsiz: Optional[np.ndarray] = None,
        fmultiple: float = 2.0,
        printrep: bool = True,
        nprint: int = 50,
        verbose: bool = True,
        save_path: Optional[str] = None) -> SGAResult:
    """
    Simple Genetic Algorithm for bounded optimization.
    
    Solves: max f(x)  subject to  lb <= x <= ub
    
    Parameters
    ----------
    func : Callable
        Objective function f(x) -> float to maximize
    lb : np.ndarray
        Lower bounds on parameters
    ub : np.ndarray
        Upper bounds on parameters
    popsize : int
        Population size (must be even)
    maxgen : int
        Maximum number of generations
    lparam : np.ndarray, optional
        Bits per parameter (default: 10 for each)
    pcrossover : float
        Probability of crossover [0, 1]
    pmutation : float
        Probability of mutation per bit [0, 1]
    gradchk : bool
        Whether to compute and check gradient
    gtol : float
        Gradient tolerance for optimality
    agflag : bool
        True if func returns (f, gradient) tuple
    side : np.ndarray, optional
        1 for one-sided, 2 for two-sided finite differences
    typsiz : np.ndarray, optional
        Typical size of parameters
    fmultiple : float
        Fitness scaling multiplier
    printrep : bool
        Whether to print reports
    nprint : int
        Print report every nprint generations
    verbose : bool
        Print progress messages
    save_path : str, optional
        Path to save intermediate results
        
    Returns
    -------
    SGAResult
        Optimization results
    """
    nparam = len(lb)
    
    # Validate inputs
    if popsize % 2 != 0:
        raise ValueError("popsize must be even")
    
    # Set defaults
    if lparam is None:
        lparam = 10 * np.ones(nparam, dtype=int)
    if side is None:
        side = 2 * np.ones(nparam, dtype=int)
    if typsiz is None:
        typsiz = np.ones(nparam)
    
    typsiz = np.abs(typsiz)
    lchrom = int(np.sum(lparam))
    
    # Initialize population randomly
    oldpopc = (np.random.rand(popsize, lchrom) < 0.5).astype(float)
    oldpopx = decode(oldpopc, nparam, lparam, ub, lb)
    
    oldpopo = np.zeros(popsize)
    for j in range(popsize):
        oldpopo[j] = func(oldpopx[j, :])
    
    # Initial statistics
    maxf, minf, avgf, sumfitness, best = statistics(oldpopo, popsize)
    xbest = oldpopx[best, :].copy()
    fbest = oldpopo[best]
    
    # Compute initial gradient
    gbest = np.zeros(nparam)
    if gradchk:
        if agflag:
            _, gbest = func(xbest)
        else:
            gbest = numerical_gradient(func, xbest, typsiz, side)
        
        if np.linalg.norm(gbest) < gtol * (1 + abs(fbest)):
            return SGAResult(xbest, fbest, gbest, code=1, generations=0)
    
    # Scale fitness if negative values present
    if minf < 0:
        oldpopf, sumfitness = scale_population(oldpopo, maxf, minf, avgf, fmultiple)
    else:
        oldpopf = oldpopo.copy()
    
    # Print initial report
    if verbose:
        print()
        print("-" * 80)
        print("         A SIMPLE GENETIC ALGORITHM FOR OPTIMIZATION")
        print("-" * 80)
        print()
        print("  SGA Parameters")
        print("  --------------")
        print()
        print(f"  Population Size (popsize)          = {popsize}")
        print(f"  Number of Elements in x (nparam)   = {nparam}")
        for j in range(nparam):
            print(f"    Chromosome Length, Max and Min Values for Parameter {j+1}: "
                  f"{lparam[j]}, {ub[j]}, {lb[j]}")
        print(f"  Chromosome Length (lchrom)         = {lchrom}")
        print(f"  Maximum Generations (maxgen)       = {maxgen}")
        print(f"  Crossover Probability (pcrossover) = {pcrossover}")
        print(f"  Mutation Probability (pmutation)   = {pmutation}")
        print()
        print("  Initial Generation Statistics")
        print("  -----------------------------")
        print()
        print(f"  Initial population maximum fitness = {maxf:g}")
        print(f"  Initial population minimum fitness = {minf:g}")
        print(f"  Initial population average fitness = {avgf:g}")
        print(f"  Initial population sum of fitness  = {sumfitness:g}")
        print()
    
    # Main evolution loop
    ncross = 0
    nmutation = 0
    
    newpopc = np.zeros((popsize, lchrom))
    newpopx = np.zeros((popsize, nparam))
    newpopo = np.zeros(popsize)
    newpopp1 = np.zeros(popsize, dtype=int)
    newpopp2 = np.zeros(popsize, dtype=int)
    newpopxs = np.zeros(popsize, dtype=int)
    
    for gen in range(1, maxgen + 1):
        j = 0
        while j < popsize:
            # Select parents
            mate1 = select(popsize, sumfitness, oldpopf)
            mate2 = select(popsize, sumfitness, oldpopf)
            
            # Determine crossover point
            if np.random.rand() < pcrossover:
                jcross = 1 + int(np.floor((lchrom - 1) * np.random.rand()))
                ncross += 1
            else:
                jcross = lchrom
            
            # Create offspring with crossover and mutation
            mutation1 = np.random.rand(jcross) < pmutation
            mutation2 = np.random.rand(lchrom - jcross) < pmutation
            
            newpopc[j, :jcross] = np.abs(oldpopc[mate1, :jcross] - mutation1.astype(float))
            newpopc[j, jcross:] = np.abs(oldpopc[mate2, jcross:] - mutation2.astype(float))
            
            mutation3 = np.random.rand(jcross) < pmutation
            mutation4 = np.random.rand(lchrom - jcross) < pmutation
            
            newpopc[j+1, :jcross] = np.abs(oldpopc[mate2, :jcross] - mutation3.astype(float))
            newpopc[j+1, jcross:] = np.abs(oldpopc[mate1, jcross:] - mutation4.astype(float))
            
            # Decode and evaluate
            newpopx[j:j+2, :] = decode(newpopc[j:j+2, :], nparam, lparam, ub, lb)
            newpopo[j] = func(newpopx[j, :])
            newpopo[j+1] = func(newpopx[j+1, :])
            
            newpopp1[j:j+2] = mate1
            newpopp2[j:j+2] = mate2
            newpopxs[j:j+2] = jcross
            
            j += 2
        
        # Statistics for new generation
        maxf, minf, avgf, sumfitness, best = statistics(newpopo, popsize)
        
        # Update best if improved
        if newpopo[best] > fbest:
            xbest = newpopx[best, :].copy()
            fbest = newpopo[best]
            
            if verbose:
                print(f" Best objective function at generation {gen}: {fbest:g}")
            
            if gradchk:
                if agflag:
                    fbest, gbest = func(xbest)
                else:
                    gbest = numerical_gradient(func, xbest, typsiz, side)
                
                if verbose:
                    print(" Associated parameter and gradient vectors:")
                    print(np.column_stack([xbest, gbest]))
                
                if np.linalg.norm(gbest) < gtol * (1 + abs(fbest)):
                    if save_path:
                        _save_state(save_path, locals())
                    return SGAResult(xbest, fbest, gbest, code=1, generations=gen)
            else:
                if verbose:
                    print(" Associated parameter vector:")
                    print(xbest)
        
        # Scale fitness
        if minf < 0:
            newpopf, sumfitness = scale_population(newpopo, maxf, minf, avgf, fmultiple)
        else:
            newpopf = newpopo.copy()
        
        # Print periodic report
        if printrep and gen % nprint == 0 and verbose:
            print()
            print(f"  Statistics Report for Generation {gen}:")
            print(f"    maxf  = {maxf:12.4e}, minf  = {minf:12.4e}, avgf  = {avgf:12.4e}")
            print(f"    fbest = {fbest:12.4e}")
            if gradchk:
                print("    xbest =      gbest =")
                print(np.column_stack([xbest, gbest]))
            else:
                print("    xbest =")
                print(xbest)
            print("-" * 80)
            print()
        
        # Swap populations
        oldpopc = newpopc.copy()
        oldpopx = newpopx.copy()
        oldpopf = newpopf.copy()
        oldpopo = newpopo.copy()
        
        # Save intermediate state
        if save_path and gen % 20 == 0:
            _save_state(save_path, {
                'gen': gen,
                'oldpopc': oldpopc,
                'oldpopx': oldpopx,
                'oldpopf': oldpopf,
                'oldpopo': oldpopo,
                'xbest': xbest,
                'fbest': fbest,
                'gbest': gbest,
                'sumfitness': sumfitness
            })
    
    # Max iterations reached
    if save_path:
        _save_state(save_path, {
            'gen': maxgen,
            'oldpopc': oldpopc,
            'oldpopx': oldpopx,
            'oldpopf': oldpopf,
            'oldpopo': oldpopo,
            'xbest': xbest,
            'fbest': fbest,
            'gbest': gbest
        })
    
    return SGAResult(xbest, fbest, gbest, code=2, generations=maxgen)


def _save_state(path: str, state: dict):
    """Save SGA state to file."""
    with open(path, 'wb') as f:
        pickle.dump(state, f)


def load_state(path: str) -> dict:
    """Load SGA state from file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def resume_sga(func: Callable[[np.ndarray], float],
               state_path: str,
               maxgen: int,
               **kwargs) -> SGAResult:
    """
    Resume SGA optimization from saved state.
    
    Parameters
    ----------
    func : Callable
        Objective function
    state_path : str
        Path to saved state file
    maxgen : int
        New maximum generation limit
    **kwargs
        Additional parameters to override
        
    Returns
    -------
    SGAResult
        Optimization results
    """
    state = load_state(state_path)
    
    # This would require more implementation to properly resume
    # For now, return the saved best
    return SGAResult(
        xbest=state['xbest'],
        fbest=state['fbest'],
        gbest=state.get('gbest', np.zeros_like(state['xbest'])),
        code=2,
        generations=state['gen']
    )


# Example usage and testing
if __name__ == "__main__":
    from .objfunc import rosenbrock
    
    # Test on Rosenbrock function (negated for maximization)
    lb = np.array([0.0, 0.0])
    ub = np.array([2.0, 2.0])
    
    result = sga(rosenbrock, lb, ub, maxgen=100, verbose=True)
    print(f"\nOptimization complete: {result}")
    print(f"Best x: {result.xbest}")
    print(f"Best f: {result.fbest}")
