# Changelog: Companion Notebook 1

**Subject:** Change history for `jupyter/companion-notebook-1.ipynb`  
**Paper:** "Money as a Medium of Exchange in an Economy with Artificially Intelligent Agents," *Journal of Economic Dynamics and Control*, 14, 329–373.

---

## 2026-02-18

### Bug Fixes

- **Economy C GA parameters:** Fixed `ga_pcross=0.01, ga_pmutation=0.2` → `ga_pcross=0.6, ga_pmutation=0.01`. Confirmed as a bug by reading the original MATLAB code: `class003.m` calls `winitial` with no overrides, and `winitial.m` sets `pcrosst=0.6, pmutationt=0.01` universally for all economies.

### Algorithm Improvements

- **GA rewritten to replicate `ga3.m`:** Complete rewrite of `apply_genetic_algorithm()` to faithfully replicate the MATLAB `ga3.m` algorithm:
  - Two-stage parent selection: usage-weighted pre-selection ($p_2 = 0.7$), then fitness-proportional roulette wheel
  - Single-point crossover (was two-point) matching MATLAB
  - Ternary cyclic mutation (was random choice) matching MATLAB
  - Crowding-based replacement via `cankill` filtering (was weakest replacement) matching MATLAB `crowdin3.m`
  - N_pairs now computed from `round(propselect * n_classifiers * 0.5)` (7 trade pairs, 1 consume pair for 72/12 classifiers)
- **Creation operator fixed:** New `create_classifier_replacing_weakest()` replaces the most redundant or weakest classifier (matching MATLAB `create.m`), maintaining constant population size. Previously, classifiers were appended, growing the population unboundedly.
- **EconomyConfig extended:** Added GA parameters: `ga_propselect` ($p_1=0.2$), `ga_propused` ($p_2=0.7$), `ga_crowdfactor_trade` (8), `ga_crowdfactor_consume` (4), `ga_crowdsubpop` (0.5), `ga_uratio` (0.0, 0.2)
- **GA parameters propagated:** `KiyotakiWrightSimulation`, `FiatMoneySimulation`, and `FiveGoodSimulation` now pass all GA parameters through to `apply_genetic_algorithm()`

### New Features

- **A1.2 classifier strength tables:** Added `print_classifier_strengths()` calls for all 3 agent types at both t=1000 and t=2000, replicating paper Tables 20–27
- **Data-driven Wicksell triangles:** New `plot_wicksell_triangle()` function reads actual exchange counts from simulation history and draws exchange pattern diagrams with arrow thickness proportional to exchange frequency. Called for all 8 economies with appropriate layout (triangle for 3-type, pentagon for 5-type).

---

## 2026-02-17

### Bug Fixes

- **Economy D IndexError:** Fixed crash in `print_5good_exchange_freq()`, `print_5good_holdings()`, and `print_5good_winning_actions()` when called with negative `t_idx` — negative indices now correctly converted to positive before slicing

### New Features

- **Time-indexed analysis:** `print_exchange_frequency()` and `print_consumption_frequency()` now accept a `t_idx` parameter for querying exchange/consumption frequencies at any historical time point, not just the final period
- **Classifier strength tables:** Added `print_classifier_strengths()` function showing top-N classifiers with condition strings, actions, strengths, usage counts, and specificity — replicating paper Tables 10–15
- **Enhanced `print_full_analysis()`:** Now accepts `t_idx` for time-specific full-analysis snapshots; uses historical holdings distributions when querying past time points
- **Intermediate reporting standardized:** Economies A1.2, A2.1, A2.2, B.1, and B.2 now report full analysis (holdings, exchange freq, consumption freq, winning actions) at intermediate time points using `print_full_analysis()` with `t_idx`, replacing ad-hoc manual holdings-only reporting

### Algorithm Improvements

- **Diversification operator activated:** `apply_diversification()` now called in `get_trade_decision()` and `get_consume_decision()` for all three agent classes (`ClassifierAgent`, `FiatMoneyAgent`, `FiveGoodAgent`). Paper Section 6 specifies diversification is "used each time the classifier system is called upon" to ensure both actions are represented among matching classifiers
- **Specialization operator implemented:** Added `apply_specialization()` implementing the paper's Section 6 specialization operator — probabilistically converts `#` wildcards to specific bit values with frequency $f_s(t) = 1/(2\sqrt{t})$. Integrated into all three simulation loops (`KiyotakiWrightSimulation`, `FiatMoneySimulation`, `FiveGoodSimulation`)

### Documentation

- **Economy C GA parameters:** Added investigative comment noting that `ga_pcross=0.01, ga_pmutation=0.2` appear swapped vs the standard `pcross=0.6, pmutation=0.01` used in other economies — flagged for verification against the original MATLAB code

---

## 2026-02-16

### Bug Fixes

- **Classifier strength update denominators (Eqs 10–11):** Exchange classifier now uses $\tau_e$ (`n_used` before increment); consumption classifier uses $\tau_c - 1$ (`n_used` before increment) — matching the paper's exact formulation
- **Consumption of wrong goods (Eq 12):** Agents can now attempt consumption of any held good, receiving $u_i(k) = 0$ for $k \neq i$, rather than being blocked from consuming
- **Economy B.2 bid parameters:** Corrected `bid_trade` from `(0.25, 0.25)` to `(0.025, 0.025)` to match paper Table 2
- **Economy C and D:** Applied the same strength-update and consumption fixes to `FiatMoneyAgent`/`FiatMoneySimulation` and `FiveGoodAgent`/`FiveGoodSimulation`

### New Features

- **Frequency tracking:** All three simulation classes (KW, FiatMoney, FiveGood) now track exchange and consumption frequency per period
- **Analysis functions:** Added `print_exchange_frequency()`, `print_consumption_frequency()`, `print_winning_classifier_actions()`, and `print_full_analysis()` for detailed result inspection
- **Intermediate reporting:** Results now reported at time points matching the paper's tables (e.g., $t = 500, 1000$ for A1.1; $t = 1000, 2000$ for A1.2)
- **Economy-specific analysis:** Custom analysis functions for FiatMoney (4-good) and FiveGood (5-good) economies

### Housekeeping

- Fixed section numbering to be sequential (1–15)

---

## 2026-02-10

### companion-notebook-1.ipynb

**Added three missing economies to achieve complete coverage of all eight economies from the paper's master table (Table 1):**

- **Economy A2.2** (Section 5b): High utility ($u_i = 500$) with random initial classifiers and the genetic algorithm. Tests whether the GA's increased experimentation enables agents to escape the myopic lock-in that prevented the speculative equilibrium from emerging in A2.1. The paper's master table lists this economy with equilibrium type "S" (speculative).

- **Economy B.2** (Section 6b): Model B production structure with random initial classifiers and the genetic algorithm. The paper reported this economy "had not converged after 2000 periods" but was "moving towards the fundamental equilibrium."

- **Economy D** (Section 8): Five agent types, five goods, 250 total agents — the most complex economy in the paper and its key "triumph." Includes:
  - `FiveGoodAgent` class with 3-bit encoding for 5 goods
  - `FiveGoodSimulation` class with the full production structure (Type I→Good 3, II→Good 4, III→Good 5, IV→Good 1, V→Good 2)
  - Holdings distribution table and time-series plots
  - Trading pattern analysis verifying the discovered fundamental-like Nash equilibrium
  - Production structure and exchange pattern diagrams (replicating paper Figures 10–11)
  - Discussion of why this economy matters: classifier systems discovered a Nash equilibrium the authors had not analytically derived

**Updated existing sections:**

- **Summary table** (Section 9): Now includes all 8 economies (A1.1, A1.2, A2.1, A2.2, B.1, B.2, C, D) with paper equilibrium type and simulation results
- **Conclusions** (Section 14): Expanded to reflect complete coverage, highlighting Economy D as the paper's most compelling demonstration of classifier systems as equilibrium-discovery tools
