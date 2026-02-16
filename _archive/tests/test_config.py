"""
Unit tests for configuration modules.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import SGAConfig, WicksellConfig, WicksellSimpleConfig


class TestSGAConfig:
    """Tests for SGA configuration."""
    
    def test_default_creation(self):
        """Test creating default SGA config."""
        config = SGAConfig()
        
        assert config.popsize == 50
        assert config.maxgen == 50
        assert config.pcross == 0.6
        assert config.pmutation == 0.01
    
    def test_custom_parameters(self):
        """Test creating SGA config with custom parameters."""
        config = SGAConfig(
            popsize=100,
            maxgen=200,
            pcross=0.8,
            pmutation=0.05
        )
        
        assert config.popsize == 100
        assert config.maxgen == 200
        assert config.pcross == 0.8
        assert config.pmutation == 0.05
    
    def test_derived_properties(self):
        """Test computed properties."""
        config = SGAConfig(
            lparm=np.array([10, 10, 10])
        )
        
        assert config.nparms == 3
        assert config.lchrom == 30


class TestWicksellConfig:
    """Tests for Wicksell configuration."""
    
    def test_default_creation(self):
        """Test creating default Wicksell config."""
        config = WicksellConfig()
        
        assert config.maxit == 1000
        assert config.ntypes == 3
        assert config.nclasst == 72
        assert config.nclassc == 12
    
    def test_post_init_arrays(self):
        """Test that arrays are initialized correctly."""
        config = WicksellConfig()
        
        # Check strength arrays initialized
        assert config.strengtht is not None
        assert config.strengtht.shape == (72, 3)
        
        assert config.strengthc is not None
        assert config.strengthc.shape == (12, 3)
        
        # Check runit initialized
        assert config.runit is not None
        assert len(config.runit) == 1000
    
    def test_total_agents(self):
        """Test total_agents property."""
        config = WicksellConfig(
            nagents=np.array([50, 50, 50])
        )
        
        assert config.total_agents == 150
    
    def test_encoding_lengths(self):
        """Test encoding length properties."""
        config = WicksellConfig(
            bnames=np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
        )
        
        assert config.l == 3
        assert config.lcond_trade == 6
        assert config.lcond_consume == 3


class TestWicksellSimpleConfig:
    """Tests for simple Wicksell configuration."""
    
    def test_default_creation(self):
        """Test creating default simple config."""
        config = WicksellSimpleConfig()
        
        assert config.ntypes == 3
        assert config.nagents == 50
        assert config.nclassifiers == 60
    
    def test_strength_initialization(self):
        """Test strength matrix initialization."""
        config = WicksellSimpleConfig()
        
        assert config.strength is not None
        assert config.strength.shape == (60, 3)
        assert np.all(config.strength == 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
