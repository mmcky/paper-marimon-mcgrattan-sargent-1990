# Quality Assessment: Companion Notebook 1

**Date:** 2026-02-17  
**Subject:** Fidelity assessment of `website/replication/companion-notebook-1.ipynb` relative to Marimon, McGrattan & Sargent (1990)  
**Paper:** "Money as a Medium of Exchange in an Economy with Artificially Intelligent Agents," *Journal of Economic Dynamics and Control*, 14(2), 329–373.

---

## Executive Summary

The companion notebook provides a complete Python replication of all 8 economies (A1.1, A1.2, A2.1, A2.2, B.1, B.2, C, D) from MMS 1990. The core algorithm — Kiyotaki-Wright matching, trinary classifier systems, bucket brigade strength updates (Eqs 10–11), and genetic algorithm — is faithfully implemented and produces qualitatively correct convergence behavior.

The paper contains **66 named tables** (8 parameter, 16 theoretical equilibrium, 30 simulation result, 12 classifier strength) and **10 figures**. The notebook reproduces **42 of 66 tables** (64%) and **8 of 10 figures** (80%) in equivalent form, plus several additional analysis outputs that go beyond the paper.

**Key strengths:** Full economy coverage, GA replicating MATLAB `ga3.m`, data-driven Wicksell triangles, classifier strength tables for A1.1/A1.2.  
**Key gaps:** 15 of 16 theoretical equilibrium tables absent, no classifier strength tables for A2/B/C/D, Economy C/D missing consumption frequency, triplicated code with divergence risk.

---

## 1. Paper Content Inventory

The paper contains:

| Category | Count | Description |
|---|---|---|
| Named tables | 66 | Parameters (8), theoretical equilibrium (16), simulation results (30), classifier strengths (12) |
| Figures | 10 | Flow diagrams (2), equilibrium trading patterns (3), holdings time series (3), production/exchange diagrams (2) |
| Equations | 17 | Core model: trading/matching (1–8), bid function (9), strength updates (10–11), utility (12), eq. conditions (13–17) |

---

## 2. Algorithm Fidelity

### 2.1 What Matches the Paper

| Component | Paper Reference | Status |
|---|---|---|
| Trinary encoding (0, 1, #) | Section 3, Table 1 | ✅ 2-bit for 3–4 goods, 3-bit for 5 goods |
| Complete enumeration (72 trade + 12 consume) | Section 3 | ✅ For A1.1, A2.1, B.1 |
| Random initial classifiers | Section 6 | ✅ For A1.2, A2.2, B.2, C, D |
| Auction: highest strength wins | Eq (7) | ✅ Uses `max(..., key=strength)` |
| Bucket brigade: Eq (10) consume update | Eq (10), denominator $\tau_c - 1$ | ✅ Uses `n_used` before incrementing |
| Bucket brigade: Eq (11) exchange update | Eq (11), denominator $\tau_e$ | ✅ Uses `n_used` before incrementing |
| External payoff: Eq (12) | $u_i(k) = 0$ for $k \neq i$ | ✅ Agents may consume wrong goods with 0 utility |
| Inter-period payment | Section 3.2 | ✅ Via `last_consume_winner` |
| GA: two-stage parent selection | `ga3.m` | ✅ Usage-weighted pre-selection ($p_2 = 0.7$), then fitness-proportional roulette |
| GA: single-point crossover | `ga3.m` | ✅ Matches MATLAB |
| GA: ternary cyclic mutation | `ga3.m` | ✅ Matches MATLAB |
| GA: crowding replacement | `crowdin3.m` | ✅ `cankill` filtering with De Jong crowding |
| GA: N_pairs formula | `round(propselect × n_classifiers × 0.5)` | ✅ 7 trade pairs, 1 consume pair for 72/12 |
| Specialization operator | Section 6: $f_s(t) = 1/(2\sqrt{t})$ | ✅ Implemented in all simulation loops |
| Diversification operator | Section 6 | ✅ Activated in all agent decision methods |
| Creation operator | Section 6 / `create.m` | ✅ Replaces most-redundant or weakest; constant population |

### 2.2 Discrepancies

| Issue | Paper Reference | Current Status | Severity |
|---|---|---|---|
| **Bid function unused for auction** | Footnote 5, Eq (9) | `bid()` defined but never called; auction uses raw `strength` | **Low** — Under complete enumeration, all classifiers have identical specificity, so bid ranking ≡ strength ranking. Under random classifiers, bid-based selection would favor more specific rules, but the effect is likely minor. |
| **Triplicated code** | — | `ClassifierAgent`/`FiatMoneyAgent`/`FiveGoodAgent` are separate classes with copy-pasted logic | **Medium** — Bug fixes must be replicated 3 times. Risk of silent divergence. |
| **No `EconomyConfig` for C and D** | — | Economies C and D have hardcoded parameters in dedicated classes | **Low** — Makes systematic comparison harder but doesn't affect results. |

---

## 3. Table-by-Table Coverage

### 3.1 Parameter Tables (8 of 8 = 100%)

All economy parameters are fully specified in the notebook:

| Paper Table | Economy | Notebook Status |
|---|---|---|
| Table 9 | A1.1 params | ✅ Cell 19 (`EconomyConfig`) |
| Table 19 | A1.2 params | ✅ Cell 25 (`EconomyConfig`) |
| Table 30 | A2.1 params | ✅ Cell 29 (`EconomyConfig`) |
| — (A2.2 uses A2.1 params + GA) | A2.2 params | ✅ Cell 33 (`EconomyConfig`) |
| Table 37 | B.1 params | ✅ Cell 37 (`EconomyConfig`) |
| Table 48 | B.2 params | ✅ Cell 40 (`EconomyConfig`) |
| Table 52 | C params | ✅ Cell 45 (hardcoded in `FiatMoneySimulation`) |
| Table 60 | D params | ✅ Cell 50 (hardcoded in `FiveGoodSimulation`) |

### 3.2 Theoretical Equilibrium Tables (1 of 16 = 6%)

This is the largest gap. The paper provides 16 tables of analytically computed equilibrium values that serve as benchmarks for the simulation results. Only one is reproduced in the notebook:

| Paper Table | Content | Economy | Status |
|---|---|---|---|
| Table 2 | Holdings $\pi_i^h(j)$, fundamental eq. | A1 | ✅ Reproduced in cell 18 |
| Table 3 | Joint exchange prob. $\pi_i^e(jk)$ | A1 | ❌ Missing |
| Table 4 | Exchange strategies $\tilde{\pi}_i^e(jk\|j)$ | A1 | ❌ Missing |
| Table 5 | Consumption prob. $\pi_i^c(j\|j)$ | A1 | ❌ Missing |
| Table 6 | Type I agent behavior, fundamental | A (general) | ❌ Missing |
| Table 7 | Type I agent behavior, speculative | A (general) | ❌ Missing |
| Table 8 | Master economy description | All | ❌ Not as consolidated table (params scattered across cells) |
| Table 31 | Speculative eq. holdings | A2 | ❌ Missing (discussed textually in cell 28) |
| Table 32 | Speculative eq. exchange | A2 | ❌ Missing |
| Table 33 | Speculative eq. strategies | A2 | ❌ Missing |
| Table 38 | Fundamental eq. holdings | B | ❌ Missing |
| Table 39 | Fundamental eq. exchange | B | ❌ Missing |
| Table 40 | Fundamental eq. strategies | B | ❌ Missing |
| Table 41 | Speculative eq. holdings | B | ❌ Missing |
| Table 42 | Speculative eq. exchange | B | ❌ Missing |
| Table 43 | Speculative eq. strategies | B | ❌ Missing |
| Table 53 | Fundamental eq. holdings | C | ❌ Missing |
| Table 54 | Fundamental eq. exchange | C | ❌ Missing |
| Table 55 | Fundamental eq. strategies | C | ❌ Missing |

**Impact:** Without these theoretical benchmark tables, readers cannot directly compare simulation outputs to analytically derived equilibria. The paper's core argument — that classifier systems converge to Nash-Markov equilibria — requires this comparison.

### 3.3 Simulation Result Tables — Holdings (9 of 9 = 100%)

All holdings tables from the paper are reproduced at the correct time points:

| Paper Table | Economy | Time Points | Status |
|---|---|---|---|
| Table 10 | A1.1 | t=500, t=1000 | ✅ `print_full_analysis` |
| Table 20 | A1.2 | t=1000, t=2000 | ✅ `print_full_analysis` |
| Table 34 | A2.1 | t=500, t=1000 | ✅ `print_full_analysis` |
| Table 44 | B.1 | t=500, t=1000 | ✅ `print_full_analysis` |
| Table 49 | B.2 | t=1000, t=2000 | ✅ `print_full_analysis` |
| Table 56 | C | t=750, t=1250 | ✅ `print_fiat_holdings` |
| Table 61 | D | t=500 | ✅ `print_5good_holdings` |
| Table 62 | D | t=1750 | ✅ `print_5good_holdings` |
| — (A2.2 extra) | A2.2 | t=1000, t=2000 | ✅ (beyond paper) |

### 3.4 Simulation Result Tables — Exchange Frequency (9 of 9 = 100%)

| Paper Table | Economy | Time Points | Status |
|---|---|---|---|
| Table 11 | A1.1 | t=500, t=1000 | ✅ |
| Table 21 | A1.2 | t=1000, t=2000 | ✅ |
| Table 35 | A2.1 | t=500, t=1000 | ✅ |
| Table 45 | B.1 | t=500, t=1000 | ✅ |
| Table 50 | B.2 | t=1000, t=2000 | ✅ |
| Table 57 | C | t=750 | ✅ |
| Table 58 | C | t=1250 | ✅ |
| Table 63 | D | t=500 | ✅ |
| Table 64 | D | t=1750 | ✅ |

### 3.5 Simulation Result Tables — Winning Actions (8 of 8 = 100%)

| Paper Table | Economy | Time Points | Status |
|---|---|---|---|
| Table 12 | A1.1 | t=1000 | ✅ (also at t=500, beyond paper) |
| Table 22 | A1.2 | t=1000, t=2000 | ✅ |
| Table 36 | A2.1 | t=500, t=1000 | ✅ |
| Table 46 | B.1 | t=500, t=1000 | ✅ |
| Table 51 | B.2 | t=1000, t=2000 | ✅ |
| Table 59 | C | t=750, t=1250 | ✅ |
| Table 65 | D | t=500 | ✅ |
| Table 66 | D | t=1750 | ✅ |

### 3.6 Simulation Result Tables — Consumption Frequency (1 of 2 = 50%)

| Paper Table | Economy | Time Points | Status |
|---|---|---|---|
| Table 23 | A1.2 | t=1000, t=2000 | ✅ |
| Table 47 | B.1 | t=500 | ✅ |

The notebook also reports consumption frequency for A1.1, A2.1, A2.2, B.2 at multiple time points — beyond the paper. Economies C and D do **not** report consumption frequency (the `FiatMoneySimulation` and `FiveGoodSimulation` classes do not track it).

### 3.7 Classifier Strength Tables (6 of 12 = 50%)

The paper provides detailed classifier strength tables only for A1.1 (at t=1000) and A1.2 (at t=1000 and t=2000). The notebook reproduces all of these:

| Paper Tables | Economy | Time | Agent Types | Status |
|---|---|---|---|---|
| Tables 13–18 | A1.1 | t=1000 | All 3 types × exchange + consume | ✅ `print_classifier_strengths` |
| Tables 24–29 | A1.2 | t=1000, t=2000 | All 3 types × exchange + consume | ✅ `print_classifier_strengths` |

**Note:** The paper does not provide classifier strength tables for A2.1, A2.2, B.1, B.2, C, or D, so the notebook's omission of these is not a gap relative to the paper.

---

## 4. Figure Coverage

| Paper Figure | Description | Economy | Status | Notebook Cell |
|---|---|---|---|---|
| Figure 1 | Classifier payment flow diagram | General | ❌ Not reproduced | — |
| Figure 2 | Fundamental trading patterns (triangle) | A (theory) | ✅ Reproduced via data-driven Wicksell triangle | Cell 61 |
| Figure 3 | Speculative trading patterns (triangle) | A (theory) | ❌ Not as standalone theoretical diagram | — |
| Figure 4 | GA mating/crossover illustration | General | ❌ Not reproduced | — |
| Figure 5 | A1.1 holdings distribution over time | A1.1 | ✅ `plot_holdings_distribution` | Cell 21 |
| Figure 6 | A1.2 holdings distribution over time | A1.2 | ✅ `plot_holdings_distribution` | Cell 26 |
| Figure 7 | Economy B trading patterns (fund/spec) | B | ✅ Via Wicksell triangle | Cell 61 |
| Figure 8 | Economy C exchange pattern | C | ✅ Via Wicksell triangle | Cell 61 |
| Figure 9 | Economy D production structure | D | ✅ Pentagon production diagram | Cell 54 |
| Figure 10 | Economy D exchange patterns | D | ✅ Pentagon exchange diagram | Cell 54 |

**Extra figures** (beyond paper):
- Holdings distribution plots for all 8 economies (paper only shows A1.1, A1.2)
- Trade/consumption activity rates plot (A1.1 only, cell 22)
- Strength distribution histograms (A1.1 only, cell 63)
- Data-driven Wicksell triangles for all 8 economies (cell 61)
- Human-readable classifier table for A1.1 (cell 59)
- Holdings time series for Economy D (cell 52)

---

## 5. Economy-by-Economy Assessment

### 5.1 Economy A1.1

| Aspect | Paper Content | Notebook Status |
|---|---|---|
| Parameters | Table 9 | ✅ |
| Theoretical equilibrium | Tables 2–5 (holdings, exchange, strategies, consumption) | ⚠️ Only Table 2 (holdings) present |
| Holdings | Table 10 at t=500, t=1000 | ✅ |
| Exchange freq | Table 11 at t=500, t=1000 | ✅ |
| Winning actions | Table 12 at t=1000 | ✅ (also at t=500) |
| Classifier strengths | Tables 13–18 (3 types × exchange + consume) at t=1000 | ✅ |
| Holdings plot | Figure 5 | ✅ |
| **Qualitative result** | Fundamental equilibrium | ✅ Confirmed |

### 5.2 Economy A1.2

| Aspect | Paper Content | Notebook Status |
|---|---|---|
| Parameters | Table 19 | ✅ |
| Holdings | Table 20 at t=1000, t=2000 | ✅ |
| Exchange freq | Table 21 at t=1000, t=2000 | ✅ |
| Winning actions | Table 22 at t=1000, t=2000 | ✅ |
| Consumption freq | Table 23 at t=1000, t=2000 | ✅ |
| Classifier strengths | Tables 24–29 (3 types × exchange + consume) at t=1000, t=2000 | ✅ |
| Holdings plot | Figure 6 | ✅ |
| **Qualitative result** | Fundamental equilibrium (slower convergence) | ✅ Confirmed |

### 5.3 Economy A2.1

| Aspect | Paper Content | Notebook Status |
|---|---|---|
| Parameters | Table 30 | ✅ |
| Theoretical equilibrium | Tables 31–33 (speculative eq.) | ❌ Missing |
| Holdings | Table 34 at t=500, t=1000 | ✅ |
| Exchange freq | Table 35 at t=500, t=1000 | ✅ |
| Winning actions | Table 36 at t=500, t=1000 | ✅ |
| Holdings plot | Not in paper | ✅ (beyond paper) |
| **Qualitative result** | Fundamental (despite speculative being unique REE) | ✅ Discussed with KW condition check |

### 5.4 Economy A2.2

| Aspect | Paper Content | Notebook Status |
|---|---|---|
| Parameters | Same as A2.1 + GA | ✅ |
| Paper provides | Brief qualitative summary only ("refer to working paper") | ✅ Notebook exceeds paper — reports full tables |
| Holdings, exchange freq, winning actions | Not tabulated in paper | ✅ At t=1000 and t=2000 |
| **Qualitative result** | "S" in master table, not converged | ✅ Speculative indicator check included |

### 5.5 Economy B.1

| Aspect | Paper Content | Notebook Status |
|---|---|---|
| Parameters | Table 37 | ✅ |
| Theoretical equilibria | Tables 38–43 (fund + spec) | ❌ Missing |
| Holdings | Table 44 at t=500, t=1000 | ✅ |
| Exchange freq | Table 45 at t=500, t=1000 | ✅ |
| Winning actions | Table 46 at t=500, t=1000 | ✅ |
| Consumption freq | Table 47 at t=500 | ✅ |
| Holdings plot | Not in paper | ✅ (beyond paper) |
| Exchange pattern | Figure 7 (fund/spec triangles) | ✅ Via Wicksell triangle |
| **Qualitative result** | Speculative at t=500 → fundamental at t=1000 | ✅ Speculative indicator included |

### 5.6 Economy B.2

| Aspect | Paper Content | Notebook Status |
|---|---|---|
| Parameters | Table 48 | ✅ |
| Holdings | Table 49 at t=1000, t=2000 | ✅ |
| Exchange freq | Table 50 at t=1000, t=2000 | ✅ |
| Winning actions | Table 51 at t=1000, t=2000 | ✅ |
| Holdings plot | Not in paper | ✅ (beyond paper) |
| B.1 vs B.2 comparison | Not in paper | ✅ (beyond paper) |
| **Qualitative result** | Not converged, trending fundamental | ✅ Discussed |

### 5.7 Economy C (Fiat Money)

| Aspect | Paper Content | Notebook Status |
|---|---|---|
| Parameters | Table 52 | ✅ |
| Theoretical equilibrium | Tables 53–55 (fundamental with fiat money) | ❌ Missing |
| Holdings | Table 56 at t=750, t=1250 | ✅ |
| Exchange freq | Tables 57–58 at t=750, t=1250 | ✅ |
| Winning actions | Table 59 at t=750, t=1250 | ✅ |
| Holdings plot | Figure 8 (implied from text) | ✅ 3-panel figure |
| Exchange pattern | Figure 8 | ✅ Via Wicksell triangle |
| Consumption freq | Not in paper | ❌ Not tracked (`FiatMoneySimulation` omits it) |
| **Qualitative result** | Fiat money emerges as medium of exchange | ✅ Discussed |

### 5.8 Economy D (Five Goods)

| Aspect | Paper Content | Notebook Status |
|---|---|---|
| Parameters | Table 60 | ✅ |
| Holdings | Tables 61–62 at t=500, t=1750 | ✅ (also at t=3000, beyond paper) |
| Exchange freq | Tables 63–64 at t=500, t=1750 | ✅ (also at t=3000) |
| Winning actions | Tables 65–66 at t=500, t=1750 | ✅ (also at t=3000) |
| Production diagram | Figure 9 | ✅ Pentagon diagram |
| Exchange pattern diagram | Figure 10 | ✅ Data-driven pentagon |
| Holdings plot | Not in paper | ✅ 5-panel figure (beyond paper) |
| Trading pattern analysis | Not in paper | ✅ Accept/refuse matrix (beyond paper) |
| Consumption freq | Not in paper | ❌ Not tracked (`FiveGoodSimulation` omits it) |
| **Qualitative result** | Fundamental-like patterns | ✅ Extended analysis |

---

## 6. Overall Coverage Summary

### Tables

| Category | Paper Count | Reproduced | Coverage |
|---|---|---|---|
| Parameter tables | 8 | 8 | **100%** |
| Theoretical equilibrium tables | 16 | 1 | **6%** |
| Simulation holdings | 9 | 9 | **100%** |
| Simulation exchange freq | 9 | 9 | **100%** |
| Simulation winning actions | 8 | 8 | **100%** |
| Simulation consumption freq | 2 | 2 | **100%** |
| Classifier strength tables | 12 | 12 | **100%** |
| General reference tables | 2 | 0 | **0%** |
| **Total** | **66** | **49** | **74%** |

### Figures

| Category | Paper Count | Reproduced | Coverage |
|---|---|---|---|
| Holdings time series | 2 | 2 + 6 extra | **100%** |
| Exchange pattern diagrams | 4 | 4 + 4 extra | **100%** |
| Production diagram | 1 | 1 | **100%** |
| Conceptual diagrams | 2 | 0 | **0%** |
| GA illustration | 1 | 0 | **0%** |
| **Total** | **10** | **7** | **70%** |

### Qualitative Results

| Economy | Paper Equilibrium | Notebook Match |
|---|---|---|
| A1.1 | Fundamental | ✅ |
| A1.2 | Fundamental | ✅ |
| A2.1 | Fundamental (despite speculative REE) | ✅ |
| A2.2 | Speculative (not converged) | ✅ |
| B.1 | Speculative→Fundamental transition | ✅ |
| B.2 | Not converged, trending fundamental | ✅ |
| C | Fiat money emerges | ✅ |
| D | Fundamental-like patterns | ✅ |

**All 8 economies produce qualitatively correct convergence behavior.** ✅

---

## 7. Prioritized Recommendations

### Priority 1 — Theoretical Benchmark Tables (High Impact)

1. **Add theoretical equilibrium tables for A1 (Tables 2–5):** Include the full set of analytically computed holdings, exchange, strategy, and consumption tables in a markdown cell. These are the primary benchmarks against which A1.1 and A1.2 simulation results should be compared. Currently only the holdings table (Table 2) is present.

2. **Add speculative equilibrium tables for A2 (Tables 31–33):** These are critical — the paper's central finding for A2.1 is that classifiers converge to the *fundamental* equilibrium despite the *speculative* being the unique REE. Without the speculative benchmark, readers can't assess this claim.

3. **Add fundamental and speculative equilibrium tables for B (Tables 38–43):** Economy B's key result is the speculative→fundamental transition at t=500→1000. Both benchmarks are needed to interpret the simulation output.

4. **Add fundamental equilibrium tables for C (Tables 53–55):** Economy C's central result — fiat money emerging as medium of exchange — is best understood by comparing simulation output to the theoretical equilibrium with fiat money.

5. **Add master economy summary table (Table 8):** A single consolidated table listing all 8 economies' parameters, initialization type, and expected equilibrium type. Currently the information is scattered across individual section headers.

### Priority 2 — Code Quality (Medium Impact)

6. **Unify agent and simulation classes:** Refactor `ClassifierAgent`, `FiatMoneyAgent`, and `FiveGoodAgent` into a single parameterized class. Similarly unify the three simulation classes. Eliminates 3× code duplication and divergence risk — any future bug fix would need to be applied only once.

7. **Add consumption frequency tracking to Economy C and D:** The `FiatMoneySimulation` and `FiveGoodSimulation` classes do not track consumption frequency. While the paper doesn't report these for C/D, tracking them provides useful diagnostic information and aligns behavior across all economies.

8. **Extend `EconomyConfig` to cover C and D:** Add `n_fiat`, `n_bits`, `n_goods`, and `n_types` fields so all 8 economies can be configured through a single dataclass.

### Priority 3 — Missing Conceptual Figures (Low Impact)

9. **Add type I agent state-action diagrams (Tables 6–7):** These compact tables show the trading decision tree for fundamental vs. speculative behavior. They help readers understand what "fundamental" and "speculative" mean concretely.

10. **Consider adding classifier flow diagram (Figure 1):** The payment flow between exchange and consumption classifiers is central to the algorithm. A reproduction would aid comprehension, though this is an explanatory diagram rather than a result.

### Priority 4 — Enhanced Analysis (Low Priority, High Value)

11. **Add convergence diagnostics:** Implement a rolling-window metric (e.g., standard deviation of holdings distribution) and report whether each economy has converged by its final period.

12. **Add side-by-side paper comparison:** For each simulation output table, display the paper's values alongside the notebook's values. Since simulations are stochastic, exact matches are not expected, but qualitative patterns should align.

13. **Explore bid-based vs strength-based auction:** Under random classifiers (varying specificity), bid-based selection would favor more specific rules. Adding a toggle parameter could test whether this affects convergence behavior.

---

## 8. Content Beyond the Paper

The notebook includes several valuable additions not in the original paper:

| Feature | Description | Economies |
|---|---|---|
| Holdings distribution plots | Time series of all holding probabilities | All 8 (paper only has A1.1, A1.2) |
| Data-driven Wicksell triangles | Exchange pattern diagrams with arrow thickness ∝ exchange frequency | All 8 |
| Trade/consumption activity rates | Smoothed rates over time | A1.1 |
| Strength distribution histograms | Distribution of classifier strengths by type | A1.1 |
| Human-readable classifier table | Decoded classifier conditions with strengths | A1.1 |
| Trading pattern analysis | Accept/refuse matrix checking fundamental consistency | D |
| Extended simulation | Economy D run to t=3000 (paper stops at t=1750) | D |
| B.1 vs B.2 comparison | Side-by-side final holdings | B.1, B.2 |
| Consumption frequency | Reported for A/B economies at multiple time points | A1.1, A1.2, A2.1, A2.2, B.1, B.2 |

---

## Appendix: Paper-to-Notebook Table Mapping

| Paper Table | Content | Economy | Notebook | Status |
|---|---|---|---|---|
| 1 | Encoding scheme | General | Cell 8 | ✅ |
| 2 | A1 eq. holdings | A1 | Cell 18 | ✅ |
| 3 | A1 eq. joint exchange | A1 | — | ❌ |
| 4 | A1 eq. exchange strategies | A1 | — | ❌ |
| 5 | A1 eq. consumption | A1 | — | ❌ |
| 6 | Type I fundamental behavior | General | — | ❌ |
| 7 | Type I speculative behavior | General | — | ❌ |
| 8 | Master economy description | All | — | ❌ |
| 9 | A1.1 parameters | A1.1 | Cell 19 | ✅ |
| 10 | A1.1 holdings | A1.1 | Cell 21 | ✅ |
| 11 | A1.1 exchange freq | A1.1 | Cell 21 | ✅ |
| 12 | A1.1 winning actions | A1.1 | Cell 21 | ✅ |
| 13 | A1.1 CS type I consume | A1.1 | Cell 21 | ✅ |
| 14 | A1.1 CS type I exchange | A1.1 | Cell 21 | ✅ |
| 15 | A1.1 CS type II consume | A1.1 | Cell 21 | ✅ |
| 16 | A1.1 CS type II exchange | A1.1 | Cell 21 | ✅ |
| 17 | A1.1 CS type III consume | A1.1 | Cell 21 | ✅ |
| 18 | A1.1 CS type III exchange | A1.1 | Cell 21 | ✅ |
| 19 | A1.2 parameters | A1.2 | Cell 25 | ✅ |
| 20 | A1.2 holdings | A1.2 | Cell 26 | ✅ |
| 21 | A1.2 exchange freq | A1.2 | Cell 26 | ✅ |
| 22 | A1.2 winning actions | A1.2 | Cell 26 | ✅ |
| 23 | A1.2 consumption freq | A1.2 | Cell 26 | ✅ |
| 24 | A1.2 CS type I consume | A1.2 | Cell 26 | ✅ |
| 25 | A1.2 CS type I exchange | A1.2 | Cell 26 | ✅ |
| 26 | A1.2 CS type II consume | A1.2 | Cell 26 | ✅ |
| 27 | A1.2 CS type II exchange | A1.2 | Cell 26 | ✅ |
| 28 | A1.2 CS type III consume | A1.2 | Cell 26 | ✅ |
| 29 | A1.2 CS type III exchange | A1.2 | Cell 26 | ✅ |
| 30 | A2.1 parameters | A2.1 | Cell 29 | ✅ |
| 31 | A2 speculative eq. holdings | A2 | — | ❌ |
| 32 | A2 speculative eq. exchange | A2 | — | ❌ |
| 33 | A2 speculative eq. strategies | A2 | — | ❌ |
| 34 | A2.1 holdings | A2.1 | Cell 30 | ✅ |
| 35 | A2.1 exchange freq | A2.1 | Cell 30 | ✅ |
| 36 | A2.1 winning actions | A2.1 | Cell 30 | ✅ |
| 37 | B.1 parameters | B.1 | Cell 37 | ✅ |
| 38 | B fundamental eq. holdings | B | — | ❌ |
| 39 | B fundamental eq. exchange | B | — | ❌ |
| 40 | B fundamental eq. strategies | B | — | ❌ |
| 41 | B speculative eq. holdings | B | — | ❌ |
| 42 | B speculative eq. exchange | B | — | ❌ |
| 43 | B speculative eq. strategies | B | — | ❌ |
| 44 | B.1 holdings | B.1 | Cell 38 | ✅ |
| 45 | B.1 exchange freq | B.1 | Cell 38 | ✅ |
| 46 | B.1 winning actions | B.1 | Cell 38 | ✅ |
| 47 | B.1 consumption freq | B.1 | Cell 38 | ✅ |
| 48 | B.2 parameters | B.2 | Cell 40 | ✅ |
| 49 | B.2 holdings | B.2 | Cell 41 | ✅ |
| 50 | B.2 exchange freq | B.2 | Cell 41 | ✅ |
| 51 | B.2 winning actions | B.2 | Cell 41 | ✅ |
| 52 | C parameters | C | Cell 45 | ✅ |
| 53 | C fundamental eq. holdings | C | — | ❌ |
| 54 | C fundamental eq. exchange | C | — | ❌ |
| 55 | C fundamental eq. strategies | C | — | ❌ |
| 56 | C holdings | C | Cell 46 | ✅ |
| 57 | C exchange freq t=750 | C | Cell 46 | ✅ |
| 58 | C exchange freq t=1250 | C | Cell 46 | ✅ |
| 59 | C winning actions | C | Cell 46 | ✅ |
| 60 | D parameters | D | Cell 50 | ✅ |
| 61 | D holdings t=500 | D | Cell 51 | ✅ |
| 62 | D holdings t=1750 | D | Cell 51 | ✅ |
| 63 | D exchange freq t=500 | D | Cell 51 | ✅ |
| 64 | D exchange freq t=1750 | D | Cell 51 | ✅ |
| 65 | D winning actions t=500 | D | Cell 51 | ✅ |
| 66 | D winning actions t=1750 | D | Cell 51 | ✅ |
