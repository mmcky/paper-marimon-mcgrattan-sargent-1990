"""
Unit tests for utility modules.
"""

import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from decode import decode
from selection import select, select_pair
from statistics import statistics
from objfunc import objfunc
from scaling import scale_population, scale_strength
from crowding import crowding_v1, crowding_v2, crowding_v3


class TestDecode:
    """Tests for binary to real number conversion."""
    
    def test_single_parameter(self):
        """Test decoding a single parameter."""
        # Binary 11111 = 31, with 5 bits, range [0, 1]: 31/31 = 1.0
        # decode expects 2D array (popsize × lchrom)
        popc = np.array([[1, 1, 1, 1, 1]])
        nparms = 1
        lparm = np.array([5])
        maxparm = np.array([1.0])
        minparm = np.array([0.0])
        
        result = decode(popc, nparms, lparm, maxparm, minparm)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(1.0)
    
    def test_minimum_value(self):
        """Test decoding to minimum value."""
        popc = np.array([[0, 0, 0, 0, 0]])
        nparms = 1
        lparm = np.array([5])
        maxparm = np.array([1.0])
        minparm = np.array([0.0])
        
        result = decode(popc, nparms, lparm, maxparm, minparm)
        assert result[0, 0] == pytest.approx(0.0)
    
    def test_two_parameters(self):
        """Test decoding two parameters."""
        # First 5 bits: 31 -> 1.0, Next 5 bits: 0 -> 0.0
        popc = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
        nparms = 2
        lparm = np.array([5, 5])
        maxparm = np.array([1.0, 1.0])
        minparm = np.array([0.0, 0.0])
        
        result = decode(popc, nparms, lparm, maxparm, minparm)
        assert result.shape == (1, 2)
        assert result[0, 0] == pytest.approx(1.0)
        assert result[0, 1] == pytest.approx(0.0)


class TestSelect:
    """Tests for roulette wheel selection."""
    
    def test_select_returns_valid_index(self):
        """Test that select returns a valid index."""
        np.random.seed(42)
        popsize = 10
        sumfitness = 100.0
        pop = np.ones((popsize, 5)) * (sumfitness / popsize)
        pop[:, 4] = np.arange(popsize) * (sumfitness / popsize / popsize) + 1
        
        # Create fitness array
        fitness = np.ones(popsize) * 10
        
        idx = select(popsize, sum(fitness), fitness)
        assert 0 <= idx < popsize
    
    def test_select_pair_returns_two_indices(self):
        """Test that select_pair returns two valid indices."""
        np.random.seed(42)
        popsize = 10
        fitness = np.ones(popsize) * 10
        sumfitness = sum(fitness)
        
        idx1, idx2 = select_pair(popsize, sumfitness, fitness)
        assert 0 <= idx1 < popsize
        assert 0 <= idx2 < popsize


class TestStatistics:
    """Tests for population statistics."""
    
    def test_statistics_calculation(self):
        """Test statistics returns correct values."""
        fitness = np.array([10.0, 20.0, 30.0, 40.0])
        
        # statistics returns tuple: (maxf, minf, avg, sumfitness, best_idx)
        maxf, minf, avg, sumfitness, best_idx = statistics(fitness)
        
        assert maxf == 40.0
        assert minf == 10.0
        assert avg == pytest.approx(25.0)
        assert sumfitness == 100.0
        assert best_idx == 3


class TestObjfunc:
    """Tests for objective function."""
    
    def test_rosenbrock_minimum(self):
        """Test Rosenbrock function at minimum (1, 1)."""
        # Rosenbrock minimum is at (1, 1) with value 0
        # Our version is negated for maximization
        result = objfunc(np.array([1.0, 1.0]))
        assert result == pytest.approx(0.0, abs=1e-10)
    
    def test_rosenbrock_nonminimum(self):
        """Test Rosenbrock function away from minimum."""
        result = objfunc(np.array([0.0, 0.0]))
        # At (0, 0): -((1-0)^2 + 100*(0-0)^2) = -1
        assert result == pytest.approx(-1.0)


class TestScaling:
    """Tests for fitness scaling."""
    
    def test_scale_population(self):
        """Test population scaling."""
        fitness = np.array([1.0, 2.0, 3.0, 4.0])
        maxf = 4.0
        minf = 1.0
        avgf = 2.5
        fmultiple = 2.0
        
        # scale_population returns (popf, sumfitness)
        scaled, sumfitness = scale_population(fitness, maxf, minf, avgf, fmultiple)
        
        # Scaled fitness should be >= 1 (the floor)
        assert all(scaled >= 1)
        # Sum should match
        assert sumfitness == pytest.approx(np.sum(scaled))
    
    def test_scale_strength(self):
        """Test strength scaling coefficients."""
        maxs = 3.0
        mins = 0.0
        avgs = 1.5
        smultiple = 2.0
        
        # scale_strength returns coefficients (a, b)
        a, b = scale_strength(maxs, mins, avgs, smultiple)
        
        # Check that the coefficients are valid numbers
        assert np.isfinite(a)
        assert np.isfinite(b)


class TestCrowding:
    """Tests for crowding diversity maintenance."""
    
    def test_crowding_v1_returns_valid_index(self):
        """Test crowding_v1 returns valid index."""
        np.random.seed(42)
        nclass = 10
        lcond = 6
        CS = np.zeros((nclass, lcond + 5))  # Need columns for condition + strength
        CS[:, lcond + 2] = np.arange(nclass)  # Different strengths in correct column
        
        child = np.zeros(lcond + 5)
        crowdingfactor = 3
        crowdingsubpop = 3  # Must be integer
        
        idx = crowding_v1(child, CS, crowdingfactor, crowdingsubpop, nclass, lcond)
        assert 0 <= idx < nclass
    
    def test_crowding_v3_with_eligibility(self):
        """Test crowding_v3 returns valid index."""
        np.random.seed(42)
        nclass = 10
        lcond = 6
        CS = np.zeros((nclass, lcond + 5))
        CS[:, lcond + 1] = np.arange(nclass)  # Strength in correct column for v3
        
        child = np.zeros(lcond + 5)
        crowdingfactor = 3
        crowdingsubpop = 0.3  # Proportion of population
        
        idx = crowding_v3(child, CS, crowdingfactor, crowdingsubpop, nclass, lcond)
        assert 0 <= idx < nclass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
