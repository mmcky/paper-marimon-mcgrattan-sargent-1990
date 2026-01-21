"""
Wicksell Classifier System Simulations.

Agent-based simulation of commodity money emergence using classifier systems
and genetic algorithms. Based on the Wicksell N-tangles model.

Converted from wicksell.m, wnew.m, class001-004.m by Ellen R. McGrattan (1989).

This module implements the core classifier system simulation for studying
how commodity money can emerge from decentralized trading in an economy
with heterogeneous agents.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field

from .config import WicksellConfig, WicksellSimpleConfig
from .selection import select
from .statistics import statistics
from .scaling import scale_strength
from .crowding import crowding_v3
from .create import create
from .ga import ga_v3, ga_v4


@dataclass
class AgentType:
    """
    Represents a type of agent in the Wicksell economy.
    
    Each agent type has:
    - Trade classifier system (CSt): rules for when to trade
    - Consume classifier system (CSc): rules for when to consume
    - Frequency matrices for tracking behavior statistics
    """
    type_id: int
    nclasst: int       # Number of trade classifiers
    nclassc: int       # Number of consume classifiers
    l: int             # Length of good encoding
    ntypes: int        # Total number of agent types
    nback: int = 10    # Periods for rolling statistics
    
    # Classifier systems
    # Trade: [condition (2*l), action (1), strength, #trades, #used, iteration]
    CSt: np.ndarray = field(default=None)
    # Consume: [condition (l), action (1), strength, #used, iteration]
    CSc: np.ndarray = field(default=None)
    
    # Frequency matrices for probability calculations
    freq1: np.ndarray = field(default=None)  # Transition: start good j -> end good k
    freq2: np.ndarray = field(default=None)  # Holdings: times holding good j
    freq3: np.ndarray = field(default=None)  # Meeting: j meets k
    freq4: np.ndarray = field(default=None)  # Trading: j meets k and trades
    freq5: np.ndarray = field(default=None)  # Consuming: consumes good j
    
    # Rolling storage for statistics
    freq1st: np.ndarray = field(default=None)
    freq2st: np.ndarray = field(default=None)
    freq3st: np.ndarray = field(default=None)
    freq4st: np.ndarray = field(default=None)
    freq5st: np.ndarray = field(default=None)
    
    def __post_init__(self):
        """Initialize classifier systems and frequency matrices."""
        l2 = 2 * self.l
        ntype2 = self.ntypes * self.ntypes
        
        # Initialize classifier systems if not provided
        if self.CSt is None:
            # Trade classifier: [condition (l2), action (1), strength, #trades, #used, iteration]
            self.CSt = np.zeros((self.nclasst, l2 + 5))
        if self.CSc is None:
            # Consume classifier: [condition (l), action (1), strength, #used, iteration]
            self.CSc = np.zeros((self.nclassc, self.l + 4))
        
        # Initialize frequency matrices
        if self.freq1 is None:
            self.freq1 = np.zeros((self.ntypes, self.ntypes))
        if self.freq2 is None:
            self.freq2 = np.zeros(self.ntypes)
        if self.freq3 is None:
            self.freq3 = np.zeros((self.ntypes, self.ntypes))
        if self.freq4 is None:
            self.freq4 = np.zeros((self.ntypes, self.ntypes))
        if self.freq5 is None:
            self.freq5 = np.zeros(self.ntypes)
        
        # Rolling statistics storage
        if self.freq1st is None:
            self.freq1st = np.zeros((self.nback, ntype2))
        if self.freq2st is None:
            self.freq2st = np.zeros((self.nback, self.ntypes))
        if self.freq3st is None:
            self.freq3st = np.zeros((self.nback, ntype2))
        if self.freq4st is None:
            self.freq4st = np.zeros((self.nback, ntype2))
        if self.freq5st is None:
            self.freq5st = np.zeros((self.nback, self.ntypes))
    
    def roll_statistics(self):
        """Roll the statistics storage (shift and add new row)."""
        ntype2 = self.ntypes * self.ntypes
        self.freq1st = np.vstack([self.freq1st[1:, :], np.zeros(ntype2)])
        self.freq2st = np.vstack([self.freq2st[1:, :], np.zeros(self.ntypes)])
        self.freq3st = np.vstack([self.freq3st[1:, :], np.zeros(ntype2)])
        self.freq4st = np.vstack([self.freq4st[1:, :], np.zeros(ntype2)])
        self.freq5st = np.vstack([self.freq5st[1:, :], np.zeros(self.ntypes)])


def generate_random_classifiers(nclassifiers: int, lcond: int, 
                                prob: np.ndarray) -> np.ndarray:
    """
    Generate random classifier conditions.
    
    Parameters
    ----------
    nclassifiers : int
        Number of classifiers to generate
    lcond : int
        Length of condition part
    prob : np.ndarray
        2 × lcond array: prob[0,i] = P(-1), prob[1,i] = P(0) for bit i
        
    Returns
    -------
    np.ndarray
        Classifier conditions (nclassifiers × lcond) with values -1, 0, 1
    """
    # Ensure prob has 3 rows (for -1, 0, 1)
    if prob.shape[0] == 2:
        prob = np.vstack([prob, 1 - np.sum(prob, axis=0)])
    
    cs = np.cumsum(prob, axis=0)
    conditions = np.zeros((nclassifiers, lcond))
    
    for j in range(nclassifiers):
        r = np.random.rand(lcond)
        # Map random values to -1, 0, 1 based on cumulative probabilities
        conditions[j, :] = np.sum(r > cs, axis=0) - 1
    
    return conditions


def match_classifiers(CS: np.ndarray, condition: np.ndarray, 
                      lcond: int) -> np.ndarray:
    """
    Find indices of classifiers matching a condition.
    
    A classifier matches if:
    - For each bit position: classifier bit == condition bit, OR classifier bit == -1 (wildcard)
    
    Parameters
    ----------
    CS : np.ndarray
        Classifier system matrix
    condition : np.ndarray
        Condition to match (length lcond)
    lcond : int
        Length of condition
        
    Returns
    -------
    np.ndarray
        Indices of matching classifiers
    """
    nclassifiers = CS.shape[0]
    cs_cond = CS[:, :lcond]
    
    # Match: (CS >= 0 AND CS == condition) OR (CS < 0, i.e. wildcard)
    matches = np.where(cs_cond >= 0, cs_cond, condition)
    match_count = np.sum(matches == condition, axis=1)
    
    return np.where(match_count == lcond)[0]


def find_good_index(bnames: np.ndarray, good: np.ndarray) -> int:
    """Find the index of a good in the bnames encoding."""
    diffs = np.sum(np.abs(bnames - good), axis=1)
    indices = np.where(diffs == 0)[0]
    return indices[0] if len(indices) > 0 else -1


@dataclass
class SimulationState:
    """State of a classifier system simulation."""
    iteration: int = 0
    popstorage: np.ndarray = None  # Current goods held by agents
    agent_types: Dict[int, AgentType] = field(default_factory=dict)
    last: np.ndarray = None  # Last winning classifier indices
    
    # Statistics
    total_trades: int = 0
    total_consumptions: int = 0
    
    # History tracking for figures
    history: Dict[str, List] = field(default_factory=dict)
    
    def init_history(self, ntypes: int, maxit: int):
        """Initialize history storage."""
        self.history = {
            'iteration': [],
            'trades_per_iter': [],
            'consumptions_per_iter': [],
        }
        # Per-type holding distributions over time
        for t in range(ntypes):
            self.history[f'holdings_type{t}'] = []
            self.history[f'trade_prob_type{t}'] = []
    
    def record_iteration(self, ntypes: int):
        """Record statistics for current iteration."""
        self.history['iteration'].append(self.iteration)
        
        for t in range(ntypes):
            agent = self.agent_types[t]
            # Snapshot of current holding distribution
            self.history[f'holdings_type{t}'].append(agent.freq2.copy())
            # Compute trading probability matrix
            freq3 = agent.freq3 + 1e-10
            trade_prob = agent.freq4 / freq3
            self.history[f'trade_prob_type{t}'].append(trade_prob.copy())


class ClassifierSimulation:
    """
    Classifier system simulation for the Wicksell economy.
    
    This is the main simulation class that runs the agent-based model
    with classifier systems for decision making.
    """
    
    def __init__(self, config: WicksellConfig, ga_variant: str = 'ga3',
                 verbose: bool = True):
        """
        Initialize the simulation.
        
        Parameters
        ----------
        config : WicksellConfig
            Configuration parameters
        ga_variant : str
            Which GA variant to use ('ga3' or 'ga4')
        verbose : bool
            Print progress messages
        """
        self.config = config
        self.ga_variant = ga_variant
        self.verbose = verbose
        
        # Derived parameters
        self.l = config.l
        self.l2 = 2 * self.l
        self.total_agents = config.total_agents
        
        # Initialize state
        self.state = SimulationState()
        self._initialize()
    
    def _initialize(self):
        """Initialize the simulation state."""
        cfg = self.config
        
        # Create agent types
        for i in range(cfg.ntypes):
            agent = AgentType(
                type_id=i,
                nclasst=cfg.nclasst,
                nclassc=cfg.nclassc,
                l=self.l,
                ntypes=cfg.ntypes,
                nback=cfg.nback
            )
            
            # Generate random trade classifiers
            conditions = generate_random_classifiers(cfg.nclasst, self.l2, cfg.probt)
            agent.CSt[:, :self.l2] = conditions
            agent.CSt[:, self.l2] = np.round(np.random.rand(cfg.nclasst))  # Random action
            agent.CSt[:, self.l2 + 1] = cfg.strengtht[:, i]  # Initial strength
            
            # Generate random consume classifiers
            conditions = generate_random_classifiers(cfg.nclassc, self.l, cfg.probc)
            agent.CSc[:, :self.l] = conditions
            agent.CSc[:, self.l] = np.round(np.random.rand(cfg.nclassc))  # Random action
            agent.CSc[:, self.l + 1] = cfg.strengthc[:, i]  # Initial strength
            
            self.state.agent_types[i] = agent
        
        # Initialize population storage (what each agent is holding)
        self.state.popstorage = cfg.bnames[
            np.random.randint(0, cfg.ntypes, size=self.total_agents), :
        ].copy()
        
        # Initialize last winning classifiers
        max_agents = int(np.max(cfg.nagents))
        self.state.last = np.zeros((max_agents, cfg.ntypes), dtype=int)
        
        if self.verbose:
            self._print_initial_report()
    
    def _print_initial_report(self):
        """Print initial configuration report."""
        cfg = self.config
        print()
        print("Parameter Specifications")
        print("-" * 24)
        print()
        print(f" Number of agent types = {cfg.ntypes}")
        print(f" Number of trade classifiers = {cfg.nclasst}")
        print(f" Number of consume classifiers = {cfg.nclassc}")
        print(f" Maximum iterations = {cfg.maxit}")
        print()
        print(" Good labels (bnames):")
        print(cfg.bnames)
        print(" Production specialization:")
        print(cfg.produces)
        print(" Storage costs:")
        print(cfg.storecosts)
        print(" Production costs:")
        print(cfg.prodcosts)
        print(" Utility from consumption:")
        print(cfg.utility)
        print()
    
    def _get_agent_index(self, agent_num: int, type_id: int) -> int:
        """Get global index for an agent given their number and type."""
        return agent_num + int(np.sum(self.config.nagents[:type_id]))
    
    def run(self, iterations: Optional[int] = None) -> SimulationState:
        """
        Run the simulation.
        
        Parameters
        ----------
        iterations : int, optional
            Number of iterations (default: config.maxit)
            
        Returns
        -------
        SimulationState
            Final simulation state
        """
        if iterations is None:
            iterations = self.config.maxit
        
        cfg = self.config
        
        # Build agent list for random matching
        agent_list = []
        for i in range(cfg.ntypes):
            for j in range(int(cfg.nagents[i])):
                agent_list.append((j, i))  # (agent_num, type_id)
        agent_list = np.array(agent_list)
        
        # Initialize history tracking
        self.state.init_history(cfg.ntypes, iterations)
        record_interval = max(1, iterations // 100)  # Record ~100 points
        
        for it in range(iterations):
            self.state.iteration = it + 1
            
            # Roll statistics for each agent type
            for agent in self.state.agent_types.values():
                agent.roll_statistics()
            
            # Random matching
            perm = np.random.permutation(self.total_agents)
            shuffled = agent_list[perm]
            
            half = self.total_agents // 2
            mate1 = shuffled[:half]
            mate2 = shuffled[half:2*half]
            
            # Process each pair
            for i in range(half):
                self._process_match(
                    mate1[i, 0], mate1[i, 1],
                    mate2[i, 0], mate2[i, 1]
                )
            
            # Genetic algorithm (if scheduled)
            if cfg.runit[it]:
                self._run_genetic_algorithm(it + 1)
            
            # Record history at intervals
            if (it + 1) % record_interval == 0 or it == iterations - 1:
                self.state.record_iteration(cfg.ntypes)
            
            # Periodic reporting
            if self.verbose and (it + 1) % cfg.dhist == 0:
                self._print_statistics(it + 1)
        
        return self.state
    
    def _process_match(self, agent1_num: int, type1: int,
                       agent2_num: int, type2: int):
        """Process a meeting between two agents."""
        cfg = self.config
        
        # Get current holdings
        idx1 = self._get_agent_index(agent1_num, type1)
        idx2 = self._get_agent_index(agent2_num, type2)
        
        holding1 = self.state.popstorage[idx1, :]
        holding2 = self.state.popstorage[idx2, :]
        
        # Conditions for classifiers
        condition1 = np.concatenate([holding1, holding2])
        condition2 = np.concatenate([holding2, holding1])
        
        # Get agent types
        agent1 = self.state.agent_types[type1]
        agent2 = self.state.agent_types[type2]
        
        # Find good indices
        good1 = find_good_index(cfg.bnames, holding1)
        good2 = find_good_index(cfg.bnames, holding2)
        
        # Update meeting frequencies
        agent1.freq3[good1, good2] += 1
        agent2.freq3[good2, good1] += 1
        agent1.freq2[good1] += 1
        agent2.freq2[good2] += 1
        
        # Find matching trade classifiers
        matches1 = match_classifiers(agent1.CSt, condition1, self.l2)
        matches2 = match_classifiers(agent2.CSt, condition2, self.l2)
        
        # Create new classifier if no match
        if len(matches1) == 0:
            idx, agent1.CSt = create(agent1.CSt, cfg.nclasst, self.l2, condition1)
            agent1.CSt[idx, self.l2 + 4] = self.state.iteration
            matches1 = np.array([idx])
        
        if len(matches2) == 0:
            idx, agent2.CSt = create(agent2.CSt, cfg.nclasst, self.l2, condition2)
            agent2.CSt[idx, self.l2 + 4] = self.state.iteration
            matches2 = np.array([idx])
        
        # Select winning classifiers (highest strength among matches)
        strengths1 = agent1.CSt[matches1, self.l2 + 1]
        strengths2 = agent2.CSt[matches2, self.l2 + 1]
        
        winner1_idx = matches1[np.argmax(strengths1)]
        winner2_idx = matches2[np.argmax(strengths2)]
        
        # Get actions (1 = willing to trade, 0 = not willing)
        action1 = int(agent1.CSt[winner1_idx, self.l2])
        action2 = int(agent2.CSt[winner2_idx, self.l2])
        
        # Trade occurs if both willing
        trade_occurs = (action1 == 1) and (action2 == 1)
        
        if trade_occurs:
            # Swap holdings
            self.state.popstorage[idx1, :] = holding2
            self.state.popstorage[idx2, :] = holding1
            
            # Update trade frequencies
            agent1.freq4[good1, good2] += 1
            agent2.freq4[good2, good1] += 1
            
            # Update usage counts
            agent1.CSt[winner1_idx, self.l2 + 3] += 1
            agent2.CSt[winner2_idx, self.l2 + 3] += 1
            
            self.state.total_trades += 1
        
        # Process consumption decisions
        self._process_consumption(idx1, type1, good2 if trade_occurs else good1)
        self._process_consumption(idx2, type2, good1 if trade_occurs else good2)
    
    def _process_consumption(self, agent_idx: int, type_id: int, good_idx: int):
        """Process consumption decision for an agent."""
        cfg = self.config
        agent = self.state.agent_types[type_id]
        
        holding = self.state.popstorage[agent_idx, :]
        condition = holding
        
        # Find matching consume classifiers
        matches = match_classifiers(agent.CSc, condition, self.l)
        
        if len(matches) == 0:
            idx, agent.CSc = create(agent.CSc, cfg.nclassc, self.l, condition)
            agent.CSc[idx, self.l + 3] = self.state.iteration
            matches = np.array([idx])
        
        # Select winning classifier
        strengths = agent.CSc[matches, self.l + 1]
        winner_idx = matches[np.argmax(strengths)]
        
        action = int(agent.CSc[winner_idx, self.l])
        
        # Consume if action = 1 and holding consumable good
        if action == 1 and good_idx == type_id:  # Can only consume own type's good
            # Update consumption frequency
            agent.freq5[good_idx] += 1
            
            # Produce new good
            produces = cfg.produces[type_id] - 1  # 0-indexed
            self.state.popstorage[agent_idx, :] = cfg.bnames[produces, :]
            
            # Update strength based on utility
            reward = cfg.utility[type_id] - cfg.prodcosts[type_id]
            agent.CSc[winner_idx, self.l + 1] += reward
            
            self.state.total_consumptions += 1
        
        # Update usage count
        agent.CSc[winner_idx, self.l + 2] += 1
    
    def _run_genetic_algorithm(self, iteration: int):
        """Run genetic algorithm on classifier systems."""
        cfg = self.config
        
        ga_func = ga_v4 if self.ga_variant == 'ga4' else ga_v3
        
        for type_id, agent in self.state.agent_types.items():
            # Trade classifiers
            if np.random.rand() < cfg.psecond or type_id == 0:
                agent.CSt, _ = ga_func(
                    agent.CSt,
                    ctype=1,  # Trade classifier
                    nselect=int(cfg.propselectt * cfg.nclasst * 0.5),
                    pcross=cfg.pcrosst,
                    pmutation=cfg.pmutationt,
                    crowdingfactor=cfg.crowdfactort,
                    crowdingsubpop=cfg.crowdsubpopt,
                    nclass=cfg.nclasst,
                    lcond=self.l2,
                    last=self.state.last[:, type_id],
                    iteration=iteration,
                    propused=cfg.propmostusedt,
                    uratio=cfg.uratio,
                    verbose=False
                )
            
            # Consume classifiers
            if np.random.rand() < cfg.pthird:
                agent.CSc, _ = ga_func(
                    agent.CSc,
                    ctype=2,  # Consume classifier
                    nselect=int(cfg.propselectc * cfg.nclassc * 0.5),
                    pcross=cfg.pcrossc,
                    pmutation=cfg.pmutationc,
                    crowdingfactor=cfg.crowdfactorc,
                    crowdingsubpop=cfg.crowdsubpopc,
                    nclass=cfg.nclassc,
                    lcond=self.l,
                    last=self.state.last[:, type_id],
                    iteration=iteration,
                    propused=cfg.propmostusedc,
                    uratio=cfg.uratio,
                    verbose=False
                )
    
    def _print_statistics(self, iteration: int):
        """Print statistics for current iteration."""
        print(f"\n=== Iteration {iteration} ===")
        print(f"Total trades: {self.state.total_trades}")
        print(f"Total consumptions: {self.state.total_consumptions}")
        
        for type_id, agent in self.state.agent_types.items():
            print(f"\nType {type_id + 1}:")
            print(f"  Avg trade classifier strength: {np.mean(agent.CSt[:, self.l2 + 1]):.2f}")
            print(f"  Avg consume classifier strength: {np.mean(agent.CSc[:, self.l + 1]):.2f}")
    
    def get_transition_probabilities(self, type_id: int) -> np.ndarray:
        """
        Get transition probabilities for an agent type.
        
        prob[j, k] = probability of holding good k next period given holding j now
        """
        agent = self.state.agent_types[type_id]
        freq2 = agent.freq2 + 1e-10  # Avoid division by zero
        prob = agent.freq1 / freq2[:, np.newaxis]
        return prob
    
    def get_trading_probabilities(self, type_id: int) -> np.ndarray:
        """
        Get trading probabilities for an agent type.
        
        prob[j, k] = probability of trade when holding j and meeting someone with k
        """
        agent = self.state.agent_types[type_id]
        freq3 = agent.freq3 + 1e-10
        prob = agent.freq4 / freq3
        return prob


def run_simulation(config: Optional[WicksellConfig] = None,
                   ga_variant: str = 'ga3',
                   iterations: Optional[int] = None,
                   verbose: bool = True) -> ClassifierSimulation:
    """
    Run a Wicksell classifier system simulation.
    
    Parameters
    ----------
    config : WicksellConfig, optional
        Configuration (uses default if not provided)
    ga_variant : str
        GA variant ('ga3' or 'ga4')
    iterations : int, optional
        Number of iterations
    verbose : bool
        Print progress
        
    Returns
    -------
    ClassifierSimulation
        The completed simulation object
    """
    if config is None:
        config = WicksellConfig()
    
    sim = ClassifierSimulation(config, ga_variant=ga_variant, verbose=verbose)
    sim.run(iterations)
    
    return sim
