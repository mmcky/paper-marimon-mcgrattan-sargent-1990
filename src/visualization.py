"""
Visualization module for Wicksell economy simulations.

This module provides functions to generate figures similar to those
in the Marimon, McGrattan, Sargent (1990) paper, including:
- Distribution of holdings over time
- Exchange pattern diagrams
- Trading probability matrices
- Classifier system flow diagrams

Converted and extended from MATLAB plotting code.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.lines as mlines
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class FigureStyle:
    """Default style settings for figures."""
    figsize: Tuple[float, float] = (10, 8)
    dpi: int = 150
    font_size: int = 10
    title_size: int = 12
    line_width: float = 1.5
    colors: List[str] = None
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def plot_distribution_of_holdings(
    freq2_history: np.ndarray,
    agent_type: int,
    good_labels: List[str] = None,
    time_scale: int = 10,
    title: str = None,
    ax: plt.Axes = None,
    style: FigureStyle = None
) -> plt.Figure:
    """
    Plot distribution of holdings over time for an agent type.
    
    This recreates Figure 5 and Figure 6 from the paper showing
    the percentage of time agents of each type hold each good.
    
    Parameters
    ----------
    freq2_history : np.ndarray
        Shape (T, ngood) array of holding frequencies over time
    agent_type : int
        Agent type number (1-indexed for display)
    good_labels : list of str, optional
        Labels for each good (default: 'good 1', 'good 2', ...)
    time_scale : int
        Number of periods per time unit (default: 10)
    title : str, optional
        Figure title
    ax : plt.Axes, optional
        Axes to plot on (creates new figure if None)
    style : FigureStyle, optional
        Style settings
        
    Returns
    -------
    plt.Figure
        The figure object
    """
    if style is None:
        style = FigureStyle()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=style.dpi)
    else:
        fig = ax.figure
    
    T, ngood = freq2_history.shape
    
    # Convert to percentages
    totals = freq2_history.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1  # Avoid division by zero
    percentages = 100 * freq2_history / totals
    
    # Time axis
    time = np.arange(T) / time_scale
    
    if good_labels is None:
        good_labels = [f'good {i+1}' for i in range(ngood)]
    
    # Plot each good
    line_styles = ['-', '--', ':']
    for i in range(ngood):
        ax.plot(time, percentages[:, i], 
                linestyle=line_styles[i % len(line_styles)],
                linewidth=style.line_width,
                color=style.colors[i % len(style.colors)],
                label=good_labels[i])
    
    ax.set_xlabel(f'time (in periods of {time_scale})', fontsize=style.font_size)
    ax.set_ylabel('Percentage', fontsize=style.font_size)
    ax.set_ylim(-10, 100)
    ax.legend(loc='best', fontsize=style.font_size - 1)
    ax.grid(True, alpha=0.3)
    
    if title is None:
        title = f'Type {agent_type}'
    ax.set_title(title, fontsize=style.title_size)
    
    return fig


def plot_holdings_multi_type(
    freq2_histories: Dict[int, np.ndarray],
    good_labels: List[str] = None,
    time_scale: int = 10,
    title: str = None,
    style: FigureStyle = None
) -> plt.Figure:
    """
    Plot distribution of holdings for multiple agent types (subplots).
    
    Creates a figure like Figure 5 with one subplot per agent type.
    
    Parameters
    ----------
    freq2_histories : dict
        Dict mapping type_id to (T, ngood) arrays of frequencies
    good_labels : list of str, optional
        Labels for goods
    time_scale : int
        Periods per time unit
    title : str, optional
        Overall figure title
    style : FigureStyle, optional
        Style settings
        
    Returns
    -------
    plt.Figure
        The figure object
    """
    if style is None:
        style = FigureStyle()
    
    n_types = len(freq2_histories)
    fig, axes = plt.subplots(n_types, 1, figsize=(8, 3*n_types), dpi=style.dpi)
    
    if n_types == 1:
        axes = [axes]
    
    for idx, (type_id, freq2_history) in enumerate(freq2_histories.items()):
        plot_distribution_of_holdings(
            freq2_history, type_id + 1, good_labels, time_scale,
            title=f'Type {type_id + 1}', ax=axes[idx], style=style
        )
    
    if title:
        fig.suptitle(title, fontsize=style.title_size + 2)
    
    plt.tight_layout()
    return fig


def plot_exchange_pattern(
    trade_matrix: np.ndarray,
    good_labels: List[str] = None,
    title: str = None,
    threshold: float = 0.1,
    style: FigureStyle = None
) -> plt.Figure:
    """
    Plot exchange pattern diagram showing which goods trade for which.
    
    Creates a circular diagram like Figure 2, 3, 7, 8 from the paper
    showing trading patterns between goods.
    
    Parameters
    ----------
    trade_matrix : np.ndarray
        (ngood, ngood) matrix where [i,j] is probability of trading
        good i for good j
    good_labels : list of str, optional
        Labels for goods
    title : str, optional
        Figure title
    threshold : float
        Minimum probability to show an arrow
    style : FigureStyle, optional
        Style settings
        
    Returns
    -------
    plt.Figure
        The figure object
    """
    if style is None:
        style = FigureStyle()
    
    ngood = trade_matrix.shape[0]
    
    if good_labels is None:
        good_labels = [f'{i+1}' for i in range(ngood)]
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=style.dpi)
    
    # Position goods in a circle
    angles = np.linspace(0, 2*np.pi, ngood, endpoint=False)
    # Start from top and go clockwise
    angles = np.pi/2 - angles
    
    radius = 0.35
    positions = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
    
    # Draw nodes (circles for goods)
    node_radius = 0.08
    for i, (x, y) in enumerate(positions):
        circle = plt.Circle((x, y), node_radius, fill=True, 
                           facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, good_labels[i], ha='center', va='center', 
               fontsize=style.font_size + 2, fontweight='bold')
    
    # Draw arrows for trades
    for i in range(ngood):
        for j in range(ngood):
            if i != j and trade_matrix[i, j] > threshold:
                prob = trade_matrix[i, j]
                
                # Arrow from i to j
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                
                # Shorten arrow to not overlap with circles
                dx = x2 - x1
                dy = y2 - y1
                dist = np.sqrt(dx**2 + dy**2)
                
                # Start and end points adjusted for node radius
                shrink = node_radius + 0.02
                start_x = x1 + (shrink / dist) * dx
                start_y = y1 + (shrink / dist) * dy
                end_x = x2 - (shrink / dist) * dx
                end_y = y2 - (shrink / dist) * dy
                
                # Arrow width proportional to probability
                arrow_width = 0.5 + 2 * prob
                
                ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                           arrowprops=dict(arrowstyle='->', 
                                          connectionstyle='arc3,rad=0.1',
                                          lw=arrow_width,
                                          color=style.colors[i % len(style.colors)]))
                
                # Add probability label
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2
                # Offset label perpendicular to arrow
                offset = 0.05
                perp_x = -dy / dist * offset
                perp_y = dx / dist * offset
                ax.text(mid_x + perp_x, mid_y + perp_y, f'{prob:.2f}',
                       fontsize=style.font_size - 2, ha='center', va='center')
    
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect('equal')
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=style.title_size, pad=20)
    
    return fig


def plot_trading_probability_matrix(
    trade_prob: np.ndarray,
    good_labels: List[str] = None,
    agent_type: int = None,
    title: str = None,
    style: FigureStyle = None
) -> plt.Figure:
    """
    Plot trading probability matrix as a heatmap.
    
    Shows P(trade | holding good i, meeting agent with good j).
    
    Parameters
    ----------
    trade_prob : np.ndarray
        (ngood, ngood) matrix of trading probabilities
    good_labels : list of str, optional
        Labels for goods
    agent_type : int, optional
        Agent type number for title
    title : str, optional
        Figure title
    style : FigureStyle, optional
        Style settings
        
    Returns
    -------
    plt.Figure
        The figure object
    """
    if style is None:
        style = FigureStyle()
    
    ngood = trade_prob.shape[0]
    
    if good_labels is None:
        good_labels = [f'{i+1}' for i in range(ngood)]
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=style.dpi)
    
    im = ax.imshow(trade_prob, cmap='YlOrRd', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('P(trade)', fontsize=style.font_size)
    
    # Labels
    ax.set_xticks(range(ngood))
    ax.set_yticks(range(ngood))
    ax.set_xticklabels(good_labels)
    ax.set_yticklabels(good_labels)
    
    ax.set_xlabel('Partner holding', fontsize=style.font_size)
    ax.set_ylabel('Own holding', fontsize=style.font_size)
    
    # Add text annotations
    for i in range(ngood):
        for j in range(ngood):
            val = trade_prob[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   color=color, fontsize=style.font_size - 1)
    
    if title is None and agent_type is not None:
        title = f'Trading Probabilities - Type {agent_type}'
    if title:
        ax.set_title(title, fontsize=style.title_size)
    
    plt.tight_layout()
    return fig


def plot_classifier_flow_diagram(
    agent_type: int = 1,
    holding_good: str = '1',
    partner_good: str = '2',
    action: str = 'trade',
    style: FigureStyle = None
) -> plt.Figure:
    """
    Create a flow diagram showing classifier system decision process.
    
    This is a conceptual diagram like Figure 1 in the paper showing
    how classifiers determine trading decisions.
    
    Parameters
    ----------
    agent_type : int
        Agent type number
    holding_good : str
        Good the agent is holding
    partner_good : str
        Good the partner is holding
    action : str
        Resulting action ('trade' or 'no trade')
    style : FigureStyle, optional
        Style settings
        
    Returns
    -------
    plt.Figure
        The figure object
    """
    if style is None:
        style = FigureStyle()
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=style.dpi)
    
    # Draw boxes
    box_height = 0.15
    box_width = 0.2
    
    # Message box
    ax.add_patch(FancyBboxPatch((0.1, 0.7), box_width, box_height,
                                boxstyle="round,pad=0.02", 
                                facecolor='lightgreen', edgecolor='black'))
    ax.text(0.2, 0.775, f'Message\n({holding_good}, {partner_good})', 
            ha='center', va='center', fontsize=style.font_size)
    
    # Classifier list
    ax.add_patch(FancyBboxPatch((0.4, 0.6), box_width*1.5, box_height*2,
                                boxstyle="round,pad=0.02",
                                facecolor='lightyellow', edgecolor='black'))
    ax.text(0.55, 0.75, 'Classifier List\n(Conditions → Actions)', 
            ha='center', va='center', fontsize=style.font_size)
    
    # Match list
    ax.add_patch(FancyBboxPatch((0.4, 0.3), box_width*1.5, box_height,
                                boxstyle="round,pad=0.02",
                                facecolor='lightblue', edgecolor='black'))
    ax.text(0.55, 0.375, 'Matched Classifiers', 
            ha='center', va='center', fontsize=style.font_size)
    
    # Winner selection
    ax.add_patch(FancyBboxPatch((0.4, 0.1), box_width*1.5, box_height,
                                boxstyle="round,pad=0.02",
                                facecolor='lightsalmon', edgecolor='black'))
    ax.text(0.55, 0.175, 'Highest Strength Wins', 
            ha='center', va='center', fontsize=style.font_size)
    
    # Action output
    action_color = 'lightgreen' if action == 'trade' else 'lightcoral'
    ax.add_patch(FancyBboxPatch((0.75, 0.1), box_width, box_height,
                                boxstyle="round,pad=0.02",
                                facecolor=action_color, edgecolor='black'))
    ax.text(0.85, 0.175, f'Action:\n{action.upper()}', 
            ha='center', va='center', fontsize=style.font_size)
    
    # Draw arrows
    arrow_style = dict(arrowstyle='->', lw=2, color='gray')
    
    # Message to classifier list
    ax.annotate('', xy=(0.4, 0.75), xytext=(0.3, 0.75),
                arrowprops=arrow_style)
    
    # Classifier list to match list
    ax.annotate('', xy=(0.55, 0.45), xytext=(0.55, 0.6),
                arrowprops=arrow_style)
    ax.text(0.58, 0.52, 'Match', fontsize=style.font_size - 2)
    
    # Match list to winner
    ax.annotate('', xy=(0.55, 0.25), xytext=(0.55, 0.3),
                arrowprops=arrow_style)
    ax.text(0.58, 0.27, 'Select', fontsize=style.font_size - 2)
    
    # Winner to action
    ax.annotate('', xy=(0.75, 0.175), xytext=(0.7, 0.175),
                arrowprops=arrow_style)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f'Classifier System Flow for Type {agent_type} Agent',
                fontsize=style.title_size)
    
    return fig


def plot_convergence(
    metric_history: np.ndarray,
    metric_name: str = 'Value',
    time_scale: int = 1,
    equilibrium_value: float = None,
    title: str = None,
    ax: plt.Axes = None,
    style: FigureStyle = None
) -> plt.Figure:
    """
    Plot convergence of a metric over time.
    
    Parameters
    ----------
    metric_history : np.ndarray
        1D array of metric values over time
    metric_name : str
        Name of the metric for y-axis label
    time_scale : int
        Periods per time unit
    equilibrium_value : float, optional
        Expected equilibrium value (draws horizontal line)
    title : str, optional
        Figure title
    ax : plt.Axes, optional
        Axes to plot on
    style : FigureStyle, optional
        Style settings
        
    Returns
    -------
    plt.Figure
        The figure object
    """
    if style is None:
        style = FigureStyle()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=style.dpi)
    else:
        fig = ax.figure
    
    T = len(metric_history)
    time = np.arange(T) / time_scale
    
    ax.plot(time, metric_history, linewidth=style.line_width, 
            color=style.colors[0])
    
    if equilibrium_value is not None:
        ax.axhline(y=equilibrium_value, color='red', linestyle='--',
                  linewidth=1, label=f'Equilibrium ({equilibrium_value:.2f})')
        ax.legend()
    
    ax.set_xlabel(f'Time (periods / {time_scale})', fontsize=style.font_size)
    ax.set_ylabel(metric_name, fontsize=style.font_size)
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title, fontsize=style.title_size)
    
    return fig


def plot_mating_process(
    parent1: str = '0110100',
    parent2: str = '1010011',
    crossover_points: Tuple[int, int] = (2, 5),
    style: FigureStyle = None
) -> plt.Figure:
    """
    Illustrate the mating (crossover) process for classifiers.
    
    Creates a diagram like Figure 4 showing two-point crossover.
    
    Parameters
    ----------
    parent1 : str
        Binary string for first parent
    parent2 : str
        Binary string for second parent
    crossover_points : tuple
        (start, end) indices for crossover region
    style : FigureStyle, optional
        Style settings
        
    Returns
    -------
    plt.Figure
        The figure object
    """
    if style is None:
        style = FigureStyle()
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=style.dpi)
    
    n = len(parent1)
    c1, c2 = crossover_points
    
    # Create offspring
    offspring1 = parent1[:c1] + parent2[c1:c2] + parent1[c2:]
    offspring2 = parent2[:c1] + parent1[c1:c2] + parent2[c2:]
    
    # Draw parent 1
    y_p1 = 0.8
    ax.text(0.05, y_p1, 'Parent 1:', fontsize=style.font_size, va='center')
    for i, bit in enumerate(parent1):
        color = 'lightblue' if i < c1 or i >= c2 else 'white'
        ax.add_patch(plt.Rectangle((0.2 + i*0.08, y_p1-0.05), 0.07, 0.1,
                                   facecolor=color, edgecolor='black'))
        ax.text(0.235 + i*0.08, y_p1, bit, ha='center', va='center',
               fontsize=style.font_size)
    
    # Draw parent 2
    y_p2 = 0.6
    ax.text(0.05, y_p2, 'Parent 2:', fontsize=style.font_size, va='center')
    for i, bit in enumerate(parent2):
        color = 'lightyellow' if i < c1 or i >= c2 else 'white'
        ax.add_patch(plt.Rectangle((0.2 + i*0.08, y_p2-0.05), 0.07, 0.1,
                                   facecolor=color, edgecolor='black'))
        ax.text(0.235 + i*0.08, y_p2, bit, ha='center', va='center',
               fontsize=style.font_size)
    
    # Draw crossover lines
    line_x1 = 0.2 + c1*0.08 - 0.01
    line_x2 = 0.2 + c2*0.08 - 0.01
    ax.axvline(x=line_x1, ymin=0.35, ymax=0.9, color='red', 
               linestyle='--', linewidth=2)
    ax.axvline(x=line_x2, ymin=0.35, ymax=0.9, color='red', 
               linestyle='--', linewidth=2)
    
    # Arrow
    ax.annotate('', xy=(0.45, 0.4), xytext=(0.45, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.text(0.5, 0.45, 'Crossover', fontsize=style.font_size)
    
    # Draw offspring 1
    y_o1 = 0.3
    ax.text(0.05, y_o1, 'Offspring 1:', fontsize=style.font_size, va='center')
    for i, bit in enumerate(offspring1):
        if i < c1 or i >= c2:
            color = 'lightblue'  # From parent 1
        else:
            color = 'lightyellow'  # From parent 2
        ax.add_patch(plt.Rectangle((0.2 + i*0.08, y_o1-0.05), 0.07, 0.1,
                                   facecolor=color, edgecolor='black'))
        ax.text(0.235 + i*0.08, y_o1, bit, ha='center', va='center',
               fontsize=style.font_size)
    
    # Draw offspring 2
    y_o2 = 0.1
    ax.text(0.05, y_o2, 'Offspring 2:', fontsize=style.font_size, va='center')
    for i, bit in enumerate(offspring2):
        if i < c1 or i >= c2:
            color = 'lightyellow'  # From parent 2
        else:
            color = 'lightblue'  # From parent 1
        ax.add_patch(plt.Rectangle((0.2 + i*0.08, y_o2-0.05), 0.07, 0.1,
                                   facecolor=color, edgecolor='black'))
        ax.text(0.235 + i*0.08, y_o2, bit, ha='center', va='center',
               fontsize=style.font_size)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Two-Point Crossover in Classifier Mating', 
                fontsize=style.title_size)
    
    return fig


def save_figure(fig: plt.Figure, filepath: str, **kwargs):
    """
    Save a figure to file with sensible defaults.
    
    Parameters
    ----------
    fig : plt.Figure
        Figure to save
    filepath : str
        Output path
    **kwargs
        Additional arguments to savefig
    """
    defaults = {
        'dpi': 150,
        'bbox_inches': 'tight',
        'facecolor': 'white',
        'edgecolor': 'none'
    }
    defaults.update(kwargs)
    fig.savefig(filepath, **defaults)
    plt.close(fig)


# Convenience functions for paper figure generation

def generate_paper_figures(
    simulation,
    output_dir: str = 'paper/figures',
    economy_name: str = 'A'
):
    """
    Generate all paper-style figures from a completed simulation.
    
    Parameters
    ----------
    simulation : ClassifierSimulation
        Completed simulation object
    output_dir : str
        Directory for output figures
    economy_name : str
        Name for figure titles
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    style = FigureStyle()
    cfg = simulation.config
    good_labels = [f'good {i+1}' for i in range(cfg.ntypes)]
    
    # Figure: Distribution of holdings
    # Collect frequency histories from simulation
    freq2_histories = {}
    for type_id, agent in simulation.state.agent_types.items():
        # Get cumulative freq2 - would need to track history during simulation
        freq2_histories[type_id] = agent.freq2st
    
    fig = plot_holdings_multi_type(
        freq2_histories, good_labels,
        title=f'Distribution of Holdings - Economy {economy_name}',
        style=style
    )
    save_figure(fig, f'{output_dir}/holdings_{economy_name.lower()}.png')
    
    # Figure: Exchange patterns
    for type_id, agent in simulation.state.agent_types.items():
        trade_prob = simulation.get_trading_probabilities(type_id)
        fig = plot_exchange_pattern(
            trade_prob, good_labels,
            title=f'Exchange Pattern - Type {type_id + 1}, Economy {economy_name}',
            style=style
        )
        save_figure(fig, f'{output_dir}/exchange_type{type_id+1}_{economy_name.lower()}.png')
    
    # Figure: Trading probability matrices
    for type_id, agent in simulation.state.agent_types.items():
        trade_prob = simulation.get_trading_probabilities(type_id)
        fig = plot_trading_probability_matrix(
            trade_prob, good_labels, type_id + 1,
            title=f'Trading Probabilities - Economy {economy_name}',
            style=style
        )
        save_figure(fig, f'{output_dir}/trade_prob_type{type_id+1}_{economy_name.lower()}.png')
    
    print(f"Figures saved to {output_dir}/")
