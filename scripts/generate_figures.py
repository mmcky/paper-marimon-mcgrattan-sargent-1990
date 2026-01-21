#!/usr/bin/env python
"""
Generate figures for the Marimon, McGrattan, Sargent (1990) paper.

This script runs simulations matching the paper's parameter specifications
and generates figures showing:
- Distribution of holdings over time
- Exchange pattern diagrams
- Trading probability matrices
- Conceptual diagrams (classifier flow, mating process)

Usage:
    python generate_figures.py [--economy ECONOMY] [--output-dir DIR]

Economies:
    A: 3 goods, 3 types, fundamental equilibrium (s1=0, s2=0.1, s3=0.2)
    A_spec: Speculative equilibrium variant
    B: Different storage costs
    C: Fiat money economy
    D: 5 goods, 5 types

"""

import argparse
import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import WicksellConfig
from src.classifier_simulation import ClassifierSimulation
from src.visualization import (
    FigureStyle,
    plot_holdings_multi_type,
    plot_exchange_pattern,
    plot_trading_probability_matrix,
    plot_classifier_flow_diagram,
    plot_mating_process,
    plot_convergence,
    save_figure
)


def create_economy_A_config():
    """
    Create configuration for Economy A (fundamental equilibrium).
    
    3 types, 3 goods
    Storage costs: s1=0, s2=0.1, s3=0.2 (good 1 is money)
    150 agents total (50 of each type)
    """
    config = WicksellConfig(
        ntypes=3,
        nclasst=20,
        nclassc=10,
        maxit=1000,
        dhist=100,
    )
    
    # Binary encoding for 3 goods: [1,0,0], [0,1,0], [0,0,1]
    config.bnames = np.eye(3)
    
    # 50 agents of each type
    config.nagents = np.array([50, 50, 50])
    
    # Production: type i produces good i
    config.produces = np.array([1, 2, 3])
    
    # Storage costs: s1=0, s2=0.1, s3=0.2
    config.storecosts = np.array([0.0, 0.1, 0.2])
    
    # Production costs (same for all)
    config.prodcosts = np.array([0.1, 0.1, 0.1])
    
    # Utility from consuming own good
    config.utility = np.array([1.0, 1.0, 1.0])
    
    # GA parameters
    config.pcrosst = 0.6
    config.pcrossc = 0.3
    config.pmutationt = 0.02
    config.pmutationc = 0.02
    
    # GA schedule: run every 10 iterations
    config.runit = np.zeros(config.maxit, dtype=bool)
    config.runit[9::10] = True
    
    # Reinitialize dependent arrays after changes
    config.strengtht = np.zeros((config.nclasst, config.ntypes))
    config.strengthc = np.zeros((config.nclassc, config.ntypes))
    config.Taxt = np.ones((config.nclasst, config.ntypes))
    config.Taxc = np.ones((config.nclassc, config.ntypes))
    
    return config


def create_economy_A_spec_config():
    """
    Create configuration for Economy A (speculative equilibrium).
    
    Same as Economy A but different initial conditions to reach
    speculative equilibrium where good 2 becomes money.
    """
    config = create_economy_A_config()
    
    # Different initial classifier strengths to favor speculation
    config.strengtht = np.ones((config.nclasst, config.ntypes)) * 50
    config.strengthc = np.ones((config.nclassc, config.ntypes)) * 50
    
    return config


def create_economy_C_config():
    """
    Create configuration for Economy C (fiat money).
    
    3 types, 4 goods (including fiat money)
    Fiat money has zero storage cost but no intrinsic value
    """
    config = WicksellConfig(
        ntypes=3,
        nclasst=25,
        nclassc=12,
        maxit=1500,
        dhist=150,
    )
    
    # Binary encoding for 4 goods (3 + fiat)
    config.bnames = np.array([
        [1, 0, 0, 0],  # good 1
        [0, 1, 0, 0],  # good 2
        [0, 0, 1, 0],  # good 3
        [0, 0, 0, 1],  # fiat money
    ])
    
    config.nagents = np.array([50, 50, 50])
    config.produces = np.array([1, 2, 3])
    
    # Fiat has zero storage cost
    config.storecosts = np.array([0.1, 0.1, 0.1, 0.0])
    config.prodcosts = np.array([0.1, 0.1, 0.1])
    config.utility = np.array([1.0, 1.0, 1.0])
    
    config.runit = np.zeros(config.maxit, dtype=bool)
    config.runit[9::10] = True
    
    # Reinitialize arrays
    config.strengtht = np.zeros((config.nclasst, config.ntypes))
    config.strengthc = np.zeros((config.nclassc, config.ntypes))
    config.Taxt = np.ones((config.nclasst, config.ntypes))
    config.Taxc = np.ones((config.nclassc, config.ntypes))
    
    return config


def create_economy_D_config():
    """
    Create configuration for Economy D (5 goods, 5 types).
    """
    config = WicksellConfig(
        ntypes=5,
        nclasst=30,
        nclassc=15,
        maxit=2000,
        dhist=200,
    )
    
    config.bnames = np.eye(5)
    config.nagents = np.array([30, 30, 30, 30, 30])
    config.produces = np.array([1, 2, 3, 4, 5])
    
    # Varying storage costs
    config.storecosts = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
    config.prodcosts = np.ones(5) * 0.1
    config.utility = np.ones(5)
    
    config.runit = np.zeros(config.maxit, dtype=bool)
    config.runit[9::10] = True
    
    # Reinitialize arrays
    config.strengtht = np.zeros((config.nclasst, config.ntypes))
    config.strengthc = np.zeros((config.nclassc, config.ntypes))
    config.Taxt = np.ones((config.nclasst, config.ntypes))
    config.Taxc = np.ones((config.nclassc, config.ntypes))
    
    return config


def run_and_generate_figures(config, economy_name, output_dir, verbose=True):
    """
    Run simulation and generate all figures for an economy.
    """
    print(f"\n{'='*60}")
    print(f"Running Economy {economy_name}")
    print(f"{'='*60}")
    
    # Run simulation
    sim = ClassifierSimulation(config, ga_variant='ga3', verbose=verbose)
    sim.run()
    
    style = FigureStyle()
    
    # Create output directory
    economy_dir = os.path.join(output_dir, economy_name.lower())
    os.makedirs(economy_dir, exist_ok=True)
    
    ntypes = config.ntypes
    good_labels = [f'good {i+1}' for i in range(len(config.bnames))]
    
    # Generate holding distribution figures (like Fig 5, 6)
    print("\nGenerating holding distribution figures...")
    if sim.state.history.get('holdings_type0'):
        freq2_histories = {}
        for t in range(ntypes):
            history_list = sim.state.history[f'holdings_type{t}']
            if history_list:
                freq2_histories[t] = np.array(history_list)
        
        fig = plot_holdings_multi_type(
            freq2_histories,
            good_labels[:ntypes],  # Only goods that agents produce
            time_scale=10,
            title=f'Distribution of Holdings - Economy {economy_name}',
            style=style
        )
        save_figure(fig, os.path.join(economy_dir, 'holdings_distribution.png'))
        print(f"  Saved: {economy_dir}/holdings_distribution.png")
    
    # Generate exchange pattern figures (like Fig 2, 3, 7, 8)
    print("\nGenerating exchange pattern figures...")
    for t in range(ntypes):
        trade_prob = sim.get_trading_probabilities(t)
        fig = plot_exchange_pattern(
            trade_prob,
            good_labels[:ntypes],
            title=f'Exchange Pattern - Type {t+1} Agent, Economy {economy_name}',
            style=style
        )
        save_figure(fig, os.path.join(economy_dir, f'exchange_pattern_type{t+1}.png'))
        print(f"  Saved: {economy_dir}/exchange_pattern_type{t+1}.png")
    
    # Generate trading probability heatmaps
    print("\nGenerating trading probability heatmaps...")
    for t in range(ntypes):
        trade_prob = sim.get_trading_probabilities(t)
        fig = plot_trading_probability_matrix(
            trade_prob,
            good_labels[:ntypes],
            agent_type=t+1,
            title=f'Trading Probabilities - Type {t+1}, Economy {economy_name}',
            style=style
        )
        save_figure(fig, os.path.join(economy_dir, f'trade_prob_type{t+1}.png'))
        print(f"  Saved: {economy_dir}/trade_prob_type{t+1}.png")
    
    # Print summary statistics
    print(f"\n--- Economy {economy_name} Summary ---")
    print(f"Total trades: {sim.state.total_trades}")
    print(f"Total consumptions: {sim.state.total_consumptions}")
    print(f"Iterations: {sim.state.iteration}")
    
    for t in range(ntypes):
        agent = sim.state.agent_types[t]
        print(f"\nType {t+1} final holding distribution:")
        total = agent.freq2.sum()
        if total > 0:
            pct = 100 * agent.freq2 / total
            for g in range(ntypes):
                print(f"  Good {g+1}: {pct[g]:.1f}%")
    
    return sim


def generate_conceptual_figures(output_dir):
    """Generate conceptual diagrams that don't require simulation."""
    style = FigureStyle()
    
    concepts_dir = os.path.join(output_dir, 'concepts')
    os.makedirs(concepts_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating Conceptual Figures")
    print("="*60)
    
    # Classifier flow diagram (Fig 1)
    print("\nGenerating classifier flow diagram...")
    fig = plot_classifier_flow_diagram(
        agent_type=1,
        holding_good='1',
        partner_good='2',
        action='trade',
        style=style
    )
    save_figure(fig, os.path.join(concepts_dir, 'classifier_flow.png'))
    print(f"  Saved: {concepts_dir}/classifier_flow.png")
    
    # Mating process diagram (Fig 4)
    print("\nGenerating mating process diagram...")
    fig = plot_mating_process(
        parent1='0110100',
        parent2='1010011',
        crossover_points=(2, 5),
        style=style
    )
    save_figure(fig, os.path.join(concepts_dir, 'mating_process.png'))
    print(f"  Saved: {concepts_dir}/mating_process.png")


def main():
    parser = argparse.ArgumentParser(
        description='Generate figures for Marimon, McGrattan, Sargent (1990)'
    )
    parser.add_argument(
        '--economy', '-e',
        choices=['A', 'A_spec', 'C', 'D', 'all'],
        default='A',
        help='Economy to simulate (default: A)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='paper/figures',
        help='Output directory for figures (default: paper/figures)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress simulation progress output'
    )
    parser.add_argument(
        '--conceptual-only',
        action='store_true',
        help='Only generate conceptual diagrams (no simulation)'
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Always generate conceptual figures
    generate_conceptual_figures(args.output_dir)
    
    if args.conceptual_only:
        print("\nDone (conceptual figures only).")
        return
    
    # Map economy names to configs
    economies = {
        'A': create_economy_A_config,
        'A_spec': create_economy_A_spec_config,
        'C': create_economy_C_config,
        'D': create_economy_D_config,
    }
    
    if args.economy == 'all':
        for name, config_func in economies.items():
            config = config_func()
            run_and_generate_figures(config, name, args.output_dir, 
                                    verbose=not args.quiet)
    else:
        config = economies[args.economy]()
        run_and_generate_figures(config, args.economy, args.output_dir,
                                verbose=not args.quiet)
    
    print(f"\n{'='*60}")
    print(f"All figures saved to: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
