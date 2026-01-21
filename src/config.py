"""
Configuration dataclasses for genetic algorithms and classifier systems.

Converted from initdata.m, winitial.m, wtinit.m by Ellen R. McGrattan (1989).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SGAConfig:
    """Configuration for Simple Genetic Algorithm (SGA).
    
    Used for function optimization with binary encoding.
    """
    # Population parameters
    popsize: int = 50
    lparm: np.ndarray = field(default_factory=lambda: np.array([10, 10]))
    maxparm: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0]))
    minparm: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    
    # Evolution parameters
    maxgen: int = 50
    pcross: float = 0.6
    pmutation: float = 0.01
    fmultiple: float = 2.0
    
    # Display parameters
    printrep: int = 1
    nprint: int = 1
    reportx: np.ndarray = field(default_factory=lambda: np.array([3, 3]))
    reportf: int = 3
    ncol: int = 80
    
    # Gradient checking
    gradchk: int = 1
    agflag: int = 0
    typsiz: np.ndarray = field(default_factory=lambda: np.ones(2))
    side: int = 2
    
    @property
    def nparms(self) -> int:
        """Number of parameters being optimized."""
        return len(self.lparm)
    
    @property
    def lchrom(self) -> int:
        """Total chromosome length (sum of bits for all parameters)."""
        return int(np.sum(self.lparm))


@dataclass
class WicksellSimpleConfig:
    """Simple configuration for Wicksell classifier simulation.
    
    Used by wicksell.m - the simpler version of the simulation.
    """
    # Agent parameters
    ntypes: int = 3
    nagents: int = 50  # Scalar - same number for each type
    nclassifiers: int = 60
    
    # Type encodings and economic parameters
    bnames: np.ndarray = field(default_factory=lambda: np.array([
        [0, 0],
        [1, 0],
        [-1, 1]
    ]))
    produces: np.ndarray = field(default_factory=lambda: np.array([2, 3, 1]))
    storecosts: np.ndarray = field(default_factory=lambda: np.array([
        [0.1, 1, 4],
        [0.1, 1, 4],
        [0.1, 1, 4]
    ]))
    prodcosts: np.ndarray = field(default_factory=lambda: np.array([1, 1, 1]))
    utility: np.ndarray = field(default_factory=lambda: np.array([20, 20, 20]))
    
    # Classifier parameters
    strength: Optional[np.ndarray] = None  # nclassifiers × ntypes
    maxit: int = 90
    bid1: np.ndarray = field(default_factory=lambda: 0.1 * np.ones(3))
    bid2: np.ndarray = field(default_factory=lambda: 0.1 * np.ones(3))
    tax: np.ndarray = field(default_factory=lambda: 0.0001 * np.ones(3))
    
    # Classifier generation probabilities
    prob: np.ndarray = field(default_factory=lambda: np.array([
        [0.33, 3/12, 0.33, 3/12],
        [0.33, 7/12, 0.33, 7/12]
    ]))
    lchrom: int = 2
    
    # Display parameters
    dispclass: int = 10
    Tga: int = 10  # Genetic algorithm period
    
    # GA parameters
    proportionselect: float = 0.4
    pcross: float = 0.6
    pmutation: float = 0.05
    crowdingfactor: int = 20
    crowdingsubpop: int = 20
    smultiple: float = 2.0
    
    def __post_init__(self):
        if self.strength is None:
            self.strength = np.ones((self.nclassifiers, self.ntypes))


@dataclass  
class WicksellConfig:
    """Full configuration for Wicksell classifier simulation.
    
    Used by class001.m through class004.m - the advanced simulations.
    """
    # Simulation parameters
    maxit: int = 1000
    ntypes: int = 3
    nagents: np.ndarray = field(default_factory=lambda: np.array([50, 50, 50]))
    
    # Classifier system sizes
    nclasst: int = 72  # Trade classifiers
    nclassc: int = 12  # Consume classifiers
    rclass: int = 1    # 1 = random generation, 0 = input from rules
    
    # Type encodings (trinary)
    bnames: np.ndarray = field(default_factory=lambda: np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]))
    
    # Economic parameters
    produces: np.ndarray = field(default_factory=lambda: np.array([2, 3, 1]))
    storecosts: np.ndarray = field(default_factory=lambda: np.array([
        [0.1, 1, 20],
        [0.1, 1, 20],
        [0.1, 1, 20]
    ]))
    prodcosts: np.ndarray = field(default_factory=lambda: np.array([1, 1, 1]))
    utility: np.ndarray = field(default_factory=lambda: np.array([100, 100, 100]))
    
    # Initial strengths (will be initialized in __post_init__)
    strengtht: Optional[np.ndarray] = None  # nclasst × ntypes
    strengthc: Optional[np.ndarray] = None  # nclassc × ntypes
    
    # Bidding parameters
    tbid1: np.ndarray = field(default_factory=lambda: 0.5 * np.ones(3))
    tbid2: np.ndarray = field(default_factory=lambda: 0.5 * np.ones(3))
    cbid1: np.ndarray = field(default_factory=lambda: 0.5 * np.ones(3))
    cbid2: np.ndarray = field(default_factory=lambda: 0.5 * np.ones(3))
    
    # Taxes
    Taxt: Optional[np.ndarray] = None  # nclasst × ntypes
    Taxc: Optional[np.ndarray] = None  # nclassc × ntypes
    
    # Classifier generation probabilities
    probt: np.ndarray = field(default_factory=lambda: np.array([
        [0.33, 0.33, 0.33, 0.33, 0.33, 0.33],
        [0.33, 0.33, 0.33, 0.33, 0.33, 0.33]
    ]))
    probc: np.ndarray = field(default_factory=lambda: np.array([
        [0.33, 0.33, 0.33],
        [0.33, 0.33, 0.33]
    ]))
    
    # Display and save periods
    dclasst: int = 100
    dclassc: int = 100
    dhist: int = 20
    dprob: int = 5000
    saveh: int = 20
    savec: int = 50
    savef: int = 250
    nback: int = 10
    
    # GA parameters - trade
    propselectt: float = 0.2
    pcrosst: float = 0.6
    pmutationt: float = 0.01
    crowdfactort: int = 8
    crowdsubpopt: float = 0.5
    propmostusedt: float = 0.7
    
    # GA parameters - consume
    propselectc: float = 0.2
    pcrossc: float = 0.6
    pmutationc: float = 0.01
    crowdfactorc: int = 4
    crowdsubpopc: float = 0.5
    propmostusedc: float = 0.7
    
    # GA scheduling
    runit: Optional[np.ndarray] = None  # maxit vector of 0/1
    psecond: float = 0.33
    pthird: float = 0.33
    
    # Kill eligibility thresholds
    uratio: np.ndarray = field(default_factory=lambda: np.array([0, 0.2]))
    ufitness: float = 0.5
    
    def __post_init__(self):
        """Initialize arrays that depend on other parameters."""
        # Check if arrays need to be resized
        if self.strengtht is None or self.strengtht.shape != (self.nclasst, self.ntypes):
            self.strengtht = np.zeros((self.nclasst, self.ntypes))
        if self.strengthc is None or self.strengthc.shape != (self.nclassc, self.ntypes):
            self.strengthc = np.zeros((self.nclassc, self.ntypes))
        if self.Taxt is None or self.Taxt.shape != (self.nclasst, self.ntypes):
            self.Taxt = np.ones((self.nclasst, self.ntypes))
        if self.Taxc is None or self.Taxc.shape != (self.nclassc, self.ntypes):
            self.Taxc = np.ones((self.nclassc, self.ntypes))
        if self.runit is None or len(self.runit) != self.maxit:
            # Generate GA run schedule: probability decreases with sqrt of iteration
            pga = 1.0 / np.sqrt(np.arange(1, self.maxit // 2 + 1))
            self.runit = np.zeros(self.maxit)
            # Only run GA on even iterations
            even_indices = np.arange(1, self.maxit, 2)
            self.runit[even_indices] = pga[:len(even_indices)]
            self.runit = (self.runit > np.random.rand(self.maxit)).astype(int)
    
    @property
    def total_agents(self) -> int:
        """Total number of agents across all types."""
        return int(np.sum(self.nagents))
    
    @property
    def l(self) -> int:
        """Length of binary encoding for goods."""
        return self.bnames.shape[1]
    
    @property
    def lcond_trade(self) -> int:
        """Condition length for trade classifiers (2 * l)."""
        return 2 * self.l
    
    @property
    def lcond_consume(self) -> int:
        """Condition length for consume classifiers (l)."""
        return self.l


def create_sga_config(**kwargs) -> SGAConfig:
    """Create SGA configuration with custom parameters."""
    return SGAConfig(**kwargs)


def create_wicksell_config(**kwargs) -> WicksellConfig:
    """Create Wicksell configuration with custom parameters."""
    return WicksellConfig(**kwargs)


def create_simple_wicksell_config(**kwargs) -> WicksellSimpleConfig:
    """Create simple Wicksell configuration with custom parameters."""
    return WicksellSimpleConfig(**kwargs)


# Default configurations
DEFAULT_SGA_CONFIG = SGAConfig()
DEFAULT_WICKSELL_CONFIG = WicksellConfig()
DEFAULT_WICKSELL_SIMPLE_CONFIG = WicksellSimpleConfig()
