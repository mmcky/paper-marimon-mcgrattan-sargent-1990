# Detailed Comparison: AlphaGo-Style Notebooks

## `companion-notebook-2-alphago.ipynb` vs. Tom's `kiyotaki_wright_alphago.ipynb`

This document provides a detailed comparison of the two AlphaGo-style implementations
for the Kiyotaki-Wright economy.

---

## 1. Executive Summary

Both notebooks apply modern deep learning / RL techniques to the Kiyotaki-Wright 
monetary exchange model, replacing the Holland classifier systems from the MMS (1990) paper.
However, they take **fundamentally different algorithmic approaches**:

| | **Companion Notebook 2** | **Tom's Notebook** |
|---|---|---|
| **Core Algorithm** | MCTS + Policy-Value Networks (AlphaGo Zero) | Deep Q-Networks (DQN) |
| **Decision Rule** | Forward-looking tree search + neural net prior | Backward-looking Q-value maximization |
| **Training Paradigm** | Self-play → MCTS → Network training loop | Online experience replay + target networks |
| **Economies Tested** | A1, A2, B, C (fiat money) | A1, A1.2 (no A2, B, or C) |
| **Key Finding** | Discovers speculative equilibrium in A2 | — (A2 not tested) |
| **Lines of Code** | ~1,880 (35 cells) | ~930 (27 cells) |
| **Dependencies** | NumPy + Matplotlib only | NumPy + Matplotlib only |

**Bottom line**: Companion Notebook 2 implements the actual AlphaGo architecture (MCTS + policy-value networks),
while Tom's notebook implements DQN — a different (earlier) deep RL algorithm from the same era.
Both are valid approaches to the research question, but they represent different
points in the AlphaGo design space.

---

## 2. Algorithmic Architecture

### 2.1 Companion Notebook 2: AlphaGo Zero Architecture

Faithfully adapts the AlphaGo Zero pipeline:

1. **Combined Policy-Value Network** (`PolicyValueNetwork`)
   - 2-layer MLP with shared trunk (32, 16 hidden units)
   - Separate output heads: sigmoid (policy) + tanh (value)
   - Trained via backpropagation with combined BCE + MSE loss + L2 regularization
   - Directly corresponds to AlphaGo's dual-head network

2. **Monte Carlo Tree Search** (`MCTS` class)
   - PUCT formula for action selection: $a^* = \arg\max_a [Q(s,a) + c_{\text{puct}} \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1+N(s,a)}]$
   - Heuristic rollout policy (always consume own good, random trades)
   - Value network terminal evaluation
   - Configurable simulation count and rollout depth

3. **Self-Play Training Loop** (`train_alphago_agents`)
   - Iterative cycle: simulate economy → MCTS policy improvement → train networks
   - Epsilon schedule from 0.5 → 0.1 over iterations
   - EMA policy updates (α=0.3) for stability

4. **World Model** (`EconomyModel`)
   - Models partner trade behavior using estimated acceptance probabilities
   - Tracks goods distribution for realistic MCTS rollouts
   - Updated from simulation data each training iteration

### 2.2 Tom's Notebook: Deep Q-Network (DQN) Architecture

Implements the DQN algorithm (Mnih et al., 2015):

1. **Q-Network** (`SimpleNN`)
   - 3-layer MLP (input → 64 → 64 → output)
   - ReLU activations with linear output for Q-values
   - Adam optimizer (hand-coded with momentum terms)
   - Separate networks for trade and consume decisions

2. **Experience Replay** (`ReplayBuffer`)
   - Stores (state, action, reward, next_state, done) tuples
   - Random sampling breaks temporal correlations
   - Capacity: 10,000 transitions

3. **Target Networks** (in `DQNAgent`)
   - Separate target Q-network updated every 100 steps
   - Stabilizes the moving target problem in Q-learning

4. **ε-Greedy Exploration**
   - Epsilon decays from 1.0 → 0.05 multiplicatively (×0.9995 per step)
   - Standard exploration strategy (no MCTS)

### 2.3 Key Algorithmic Differences

| Feature | Companion NB2 (MCTS+PV) | Tom's NB (DQN) |
|---|---|---|
| **Planning** | Forward search (MCTS simulates future periods) | No planning (greedy Q-value maximization) |
| **Policy representation** | Explicit policy network P(a\|s) | Implicit via argmax Q(s,a) |
| **Value estimation** | Value network V(s) + MCTS backup | Q(s,a) with target network |
| **Training signal** | MCTS-improved policy & value targets | Bellman error from replay buffer |
| **Optimizer** | SGD | Adam (hand-coded) |
| **Network size** | 2 layers (32, 16) with dual head | 3 layers (64, 64) separate trade/consume |
| **Temporal reasoning** | Explicit rollouts (15 periods ahead) | Only through discounted Q-values (γ=0.95) |
| **Exploration** | PUCT + epsilon schedule (0.5→0.1) | ε-greedy (1.0→0.05) |

---

## 3. Neural Network Comparison

### 3.1 Architecture

**Companion NB2 — `PolicyValueNetwork`:**
```
Input(2n)  →  Dense(32)  →  ReLU  →  Dense(16)  →  ReLU  →  ┬─ Dense(1) → Sigmoid  [policy head]
                                                               └─ Dense(1) → Tanh     [value head]
```
- Parameters: ~680 (for n_goods=3, input_dim=6)
- Xavier/He initialization
- SGD optimizer with L2 regularization (λ=1e-4)

**Tom's — `SimpleNN`:**
```
Input(2n)  →  Dense(64)  →  ReLU  →  Dense(64)  →  ReLU  →  Dense(2)  [Q-values: refuse/accept]
```
- Parameters: ~4,800 (for n_goods=3, input_dim=6)
- Xavier initialization
- Adam optimizer (β1=0.9, β2=0.999, ε=1e-8)

### 3.2 Training

| Aspect | Companion NB2 | Tom's NB |
|---|---|---|
| **Training data** | MCTS-generated (state, π*, V*) triples | Experience replay transitions (s,a,r,s',done) |
| **Loss function** | BCE(policy) + MSE(value) + L2 | MSE(Q-target) only |
| **Batch size** | Full state space per iteration (~9-12 states) | 64 random transitions from replay buffer |
| **Training schedule** | 10 epochs × 40 iterations = 400 updates | 1 update per simulation period |
| **Pre-training** | Yes — 300 epochs of domain knowledge | No pre-training |
| **Learning rate** | 0.005 | 0.001 |

---

## 4. Economy Model & Simulation

### 4.1 Companion NB2

**Two-phase architecture:**
1. **Training phase** (`train_alphago_agents`)
   - Iterative self-play + MCTS + network training
   - 40 iterations, ~300 evaluation periods each
   - Takes ~40-65 seconds per economy
   
2. **Evaluation phase** (`KWAlphaGoSimulation`)
   - Runs trained agents for 1000-2000 periods
   - No exploration (pure exploitation)
   - Records full holdings distribution history

**Key features:**
- `EconomyModel` class provides world model for MCTS rollouts
- Partner acceptance probabilities estimated from simulation data
- Population goods distribution tracked and updated
- Holdings initialized with production goods (economic realism)
- Fiat money support (Economy C with 4 goods)

### 4.2 Tom's Notebook

**Single-phase architecture:**
- Training and evaluation happen simultaneously
- Agents learn online during the 2000-period simulation
- ε-greedy exploration decays as simulation progresses

**Key features:**
- `KiyotakiWrightDQN` class handles both simulation and training
- No separate economy model — agents learn directly from interactions
- Random initial holdings (not production goods)
- No fiat money support (only 3-good economies)

### 4.3 Simulation Differences

| Feature | Companion NB2 | Tom's NB |
|---|---|---|
| **Agent sharing** | 1 agent per type (shared policy) | 1 DQN per type (shared Q-networks) |
| **Initial holdings** | Production goods | Random |
| **Consumption logic** | Only consume own consumption good | Only consume own consumption good |
| **Reward structure** | u_i - s(new_good) or -s(holding) | Same |
| **Fiat money** | Supported (Economy C) | Not supported |
| **Simulation length** | 1000-2000 eval periods | 2000 periods total |

---

## 5. Experimental Coverage

### 5.1 Companion NB2

| Economy | Storage Costs | Utility | Production | Result | MMS Match? |
|---|---|---|---|---|---|
| **A1** | (0.1, 1, 20) | 100 | [1,2,0] | Fundamental | ✓ Same |
| **A2** | (0.1, 1, 20) | 500 | [1,2,0] | Speculative | ✗ Different |
| **B** | (1, 4, 9) | 100 | [2,0,1] | Mixed | ≈ Similar |
| **C** | (9, 14, 29, 0) | 100 | [1,2,0]+fiat | Fiat Money | ✓ Same |

Also includes:
- Comparative analysis tables (AlphaGo vs. MMS for all economies)
- Visual comparison bar charts
- Summary table with automatic equilibrium classification
- Detailed conclusions with historical perspective

### 5.2 Tom's Notebook

| Economy | Storage Costs | Utility | Production | Result |
|---|---|---|---|---|
| **A1** | (0.1, 1, 20) | 100 | [1,2,0] | Has stored outputs (not re-executed) |
| **A1.2** | (0.1, 1, 30) | 100 | [1,2,0] | Has stored outputs (not re-executed) |

Also includes:
- Comparison function (`plot_comparison`) for Holland vs. DQN (requires classifier sim object)
- Steady-state computation function
- Exploration rate visualization
- Analysis markdown comparing DQN vs. classifier advantages/disadvantages

**Missing from Tom's notebook:**
- Economy A2 (the critical test of fundamental vs. speculative)
- Economy B (alternative production structure)
- Economy C (fiat money)
- Actual comparison with classifier system results (comparison function exists but requires companion notebook 1 data)

---

## 6. Faithfulness to AlphaGo

### 6.1 AlphaGo's Core Components (Silver et al., 2016, 2017)

| AlphaGo Component | Companion NB2 | Tom's NB |
|---|---|---|
| **Policy network** π(a\|s) | ✅ Explicit policy head (sigmoid) | ❌ No policy network (implicit via Q) |
| **Value network** V(s) | ✅ Explicit value head (tanh) | ❌ No value network (Q serves dual role) |
| **MCTS** | ✅ Full MCTS with PUCT formula | ❌ Not implemented |
| **Self-play** | ✅ Economy simulation = self-play | ✅ Agent interactions = self-play |
| **Training from MCTS** | ✅ Networks trained on MCTS targets | ❌ Networks trained on Bellman error |
| **Combined policy-value net** | ✅ Shared trunk, two heads | ❌ Separate Q-networks |

**Assessment**: 
- Companion NB2 implements 5/6 core AlphaGo components faithfully
- Tom's notebook implements DQN (Mnih et al., 2015), which is a predecessor technology — 
  the "deep neural network" part of AlphaGo but not the "tree search" part

### 6.2 What Tom's DQN Does Have from the AlphaGo Era

While not MCTS-based, Tom's DQN does include important innovations from the same research lineage:
- **Experience replay** (Mnih et al., 2015) — also used in AlphaGo's supervised learning phase
- **Target networks** — stabilization technique used throughout deep RL
- **Neural function approximation** — the foundational deep RL contribution
- **Adam optimizer** — more sophisticated than SGD used in Companion NB2
- **3-layer network** — deeper architecture (though the problem doesn't require it)

The intro markdown in Tom's notebook explicitly frames it as "AlphaGo-style" and lists
{DQN, experience replay, target networks, ε-greedy} as the AlphaGo principles being applied.
This is partially accurate — these are components used *within* AlphaGo's training pipeline,
but the signature innovation of AlphaGo (MCTS + policy-value networks) is not present.

---

## 7. Code Quality & Documentation

### 7.1 Companion NB2

**Strengths:**
- Extensive markdown explaining each component's connection to AlphaGo
- Mathematical notation (PUCT formula, loss functions, etc.)
- Detailed docstrings with parameter descriptions
- Clear separation of training and evaluation phases
- Automatic equilibrium classification with economic reasoning
- Comprehensive comparison with MMS results

**Areas for improvement:**
- Dense code cells (~200 lines for MCTS, ~220 lines for AlphaGoAgent)
- Some complexity in the heuristic rollout policy (workaround for coordination problem)
- EMA policy smoothing adds complexity that wouldn't be needed in a perfect implementation

### 7.2 Tom's Notebook

**Strengths:**
- Clean, well-structured code
- Adam optimizer implementation is thorough and correct
- Good separation of concerns (NN, ReplayBuffer, Agent, Simulation)
- Visualization functions are comprehensive (exploration rate plot, side-by-side comparison)
- Clear hyperparameter configuration in EconomyConfig

**Areas for improvement:**
- Only tests 2 of the 4 key economies from the paper
- Missing Economy A2 (the most interesting test case)
- Missing fiat money (Economy C — the paper's key finding)
- `plot_comparison` function requires classifier system data not available in the notebook
- Conclusions claim "fundamental equilibrium emerges" but only tested one parameter set
- Title says "AlphaGo-Style" but implements DQN (which is a distinctly different algorithm)

---

## 8. Results Comparison

### 8.1 Economy A1 (the baseline test)

Both notebooks should produce similar results for A1, since it's the simplest case
with a unique fundamental equilibrium. Both predict convergence to:
- Type 1 mostly holds Good 2 (production good)
- Type 2 holds a mix of production good (Good 3) and Good 1 (medium of exchange)
- Type 3 mostly holds Good 1 (not production good — uses it as money)

**Companion NB2 results** (from execution):
```
π_i^h(j) | j=1     j=2     j=3
  i=1    | 0.020   0.980   0.000    → holds production good, trades directly
  i=2    | 0.620   0.060   0.320    → holds Good 1 as money (62%)
  i=3    | 0.820   0.180   0.000    → holds Good 1 as money (82%)
```
Classification: **Fundamental** ✓

**Tom's NB results**: Stored outputs available but not re-executed in current session.
Based on the code and hyperparameters, expected to produce similar results.

### 8.2 Economy A2 (the critical test)

**Only tested in Companion NB2.**

This is the most scientifically interesting experiment — it tests whether AlphaGo's
forward-looking MCTS can overcome the backward-looking myopia that trapped the MMS
classifier system in the fundamental equilibrium.

**Companion NB2 result**: Speculative equilibrium (Type 1 holds 36% Good 3)
- This differs from MMS and is attributed to MCTS's ability to simulate future periods

**Tom's NB**: Not tested. This is a significant omission, as Economy A2's fundamental
vs. speculative contest is the central question of the paper.

### 8.3 Economies B and C

**Only tested in Companion NB2.**
- Economy B: Mixed equilibrium
- Economy C: Fiat money emerges (~32% of holdings) — matches MMS

---

## 9. Summary Table

| Dimension | Companion NB2 | Tom's NB | Winner |
|---|---|---|---|
| **Algorithm fidelity to AlphaGo** | MCTS + Policy-Value Net | DQN (no MCTS) | Companion NB2 |
| **Experimental coverage** | 4 economies (A1, A2, B, C) | 2 economies (A1, A1.2) | Companion NB2 |
| **Novel finding** | Speculative eq. in A2 | — | Companion NB2 |
| **Fiat money** | Yes (Economy C) | No | Companion NB2 |
| **Code cleanliness** | Good but dense | Clean and modular | Tom's NB |
| **Optimizer** | SGD | Adam | Tom's NB |
| **Network depth** | 2 layers | 3 layers | Tom's NB |
| **Training stability** | EMA + pre-training (workarounds needed) | Experience replay + target networks | Comparable |
| **MMS comparison** | Detailed tables + charts | Framework only (needs classifier data) | Companion NB2 |
| **Mathematical exposition** | Extensive (PUCT, loss functions) | Moderate | Companion NB2 |

---

## 10. Recommendations for Merging

1. **Tom's notebook** could be enhanced by:
   - Adding Economy A2 (fundamental vs. speculative test)
   - Adding Economy C (fiat money)
   - Adding MCTS on top of DQN for planning (making it truly "AlphaGo-style")
   - Connecting to companion-notebook-1 for actual comparison data

2. **Companion NB2** could be improved by:
   - Adopting Adam optimizer from Tom's implementation
   - Using a 3-layer network for more capacity
   - Adding experience replay as a complement to MCTS training
   - Breaking large code cells into smaller, more modular cells

3. **Both notebooks** would benefit from:
   - Running multiple random seeds for statistical robustness
   - Reporting confidence intervals on equilibrium classifications
   - Systematic hyperparameter sensitivity analysis

---

*Generated: 9 February 2026*
