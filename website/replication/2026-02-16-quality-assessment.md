# Assessment Report: Companion Notebook 1

**Date:** 2026-02-16  
**Updated:** 2026-02-18 — Items 1–8, 11, 13 addressed in [PR #5](https://github.com/mmcky/paper-marimon-mcgrattan-sargent-1990/pull/5)  
**Subject:** Fidelity assessment of `jupyter/companion-notebook-1.ipynb` relative to Marimon, McGrattan & Sargent (1990)  
**Paper:** "Money as a Medium of Exchange in an Economy with Artificially Intelligent Agents," *Journal of Economic Dynamics and Control*, 14, 329–373.

---

## Executive Summary

The companion notebook provides a substantial Python replication of all eight economies (A1.1, A1.2, A2.1, A2.2, B.1, B.2, C, D) from the MMS 1990 paper. Core mechanics—the Kiyotaki-Wright matching framework, trinary classifier encoding, bucket brigade strength updates, and genetic algorithm—are implemented and produce qualitatively correct convergence behavior. However, several algorithmic gaps, missing outputs, and a runtime bug reduce replication fidelity. This report details the findings economy-by-economy and provides prioritized recommendations.

---

## 1. Algorithm Fidelity

### 1.1 Classifier System — What Matches the Paper

| Component | Paper Reference | Notebook Status |
|---|---|---|
| Trinary encoding (0, 1, #) | Section 3, Table 2 | **Correct** — 2-bit for 3–4 goods, 3-bit for 5 goods |
| Complete enumeration (72 trade + 12 consume) | Section 3 | **Correct** for A1.1, A2.1, B.1 |
| Random initial classifiers | Section 6 | **Correct** for A1.2, A2.2, B.2, C, D |
| Auction: highest *strength* wins | Eq (7), argmax S_e^a(t) | **Correct** — uses `max(..., key=strength)` |
| Bucket brigade: Eq (10) consume update | Eq (10), denominator τ_c − 1 | **Correct** — uses `n_used` before incrementing |
| Bucket brigade: Eq (11) exchange update | Eq (11), denominator τ_e | **Correct** — uses `n_used` before incrementing |
| External payoff: Eq (12) | u_i(k) = 0 for k ≠ i | **Correct** — agents may consume wrong goods with 0 utility |
| Exchange counter: only incremented when trade happens OR action=0 | Section 3.2 | **Correct** |
| Inter-period payment: previous consume classifier receives bid from current exchange | Section 3.2 | **Implemented** via `last_consume_winner` |

### 1.2 Classifier System — Discrepancies

| Issue | Paper Reference | Notebook Status | Severity |
|---|---|---|---|
| **Bid function unused for selection** | Footnote 5 mentions "highest bidder"; Eq (9) defines bid = (b₁ + b₂σ_e) × S | `Classifier.bid()` defined but never called for auction; selection uses raw `strength` | **Low** — Eq (7) formally uses argmax S_e, not argmax bid. Under complete enumeration all classifiers have the same specificity so bid ranking ≡ strength ranking. Under random classifiers it could matter slightly. |
| **Specialization operator not implemented** | Section 6: randomly switches # to the specific bit | ✅ **Implemented** — `apply_specialization()` with $f_s(t) = 1/(2\sqrt{t})$ added to all simulation loops | ~~**Medium**~~ **Resolved** |
| **Diversification defined but never called** | Section 6: ensures both actions represented for a state | ✅ **Activated** — `apply_diversification()` now called in all agent decision methods | ~~**Medium**~~ **Resolved** |
| **Creation operator partial** | Section 6: create new classifier when M_e(z) is empty | ✅ **Fixed** — `create_classifier_replacing_weakest()` replaces most redundant or weakest classifier, matching MATLAB `create.m`. Population size now constant. | ~~**Low**~~ **Resolved** |
| **GA parameters mapping** | Paper tables specify p₁=0.2, p₂=0.7, p₃=0.6, p₄=0.01, N_e, N_c from formula | ✅ **Refined** — GA now replicates `ga3.m`: two-stage parent selection (usage-weighted pre-selection p₂=0.7, then fitness-proportional), single-point crossover, ternary cyclic mutation, crowding-based replacement with `cankill` filtering. `EconomyConfig` extended with `ga_propselect`, `ga_propused`, `ga_crowdfactor_trade/consume`, `ga_crowdsubpop`, `ga_uratio`. N_pairs computed from `round(propselect * n_classifiers * 0.5)`. | ~~**Medium**~~ **Resolved** |
| **Economy C GA parameters anomaly** | Paper Eq (9): same bid structure for all economies | ✅ **Fixed** — Changed to `ga_pcross=0.6, ga_pmutation=0.01`, matching MATLAB `winitial.m` (universal params, no overrides in `class003.m`). The previous values (0.01/0.2) were confirmed as a bug by reading the original MATLAB code. | ~~**High**~~ **Resolved** |

### 1.3 Code Architecture Issues

| Issue | Description | Impact |
|---|---|---|
| **Triplicated agent classes** | `ClassifierAgent`, `FiatMoneyAgent`, `FiveGoodAgent` are separate classes with copy-pasted `update_strengths()` logic | Bug fixes must be replicated 3 times. Risk of silent divergence between implementations. |
| **No `EconomyConfig` for C and D** | Economies C and D bypass the `EconomyConfig` dataclass, using hardcoded parameters in dedicated simulation classes | Makes systematic parameter comparison impossible. Summary tables must be hand-coded. |
| **No shared base class** | Three simulation classes (`KiyotakiWrightSimulation`, `FiatMoneySimulation`, `FiveGoodSimulation`) duplicate matching/trading/GA loop logic | Same fix-propagation risks as triplicated agents. |

---

## 2. Economy-by-Economy Comparison

### 2.1 Economy A1.1 (Complete Enumeration, Fundamental Equilibrium)

**Parameters:** All match the paper (Table 3). ✅

| Paper Table | Content | Notebook | Status |
|---|---|---|---|
| Table 7 | Holdings π_i^h(j) at t=500, t=1000 | `print_holdings_table()` at t=500 and t=1000 | ✅ Reported |
| Table 8 | Exchange frequency π_i^e(jk) at t=500, t=1000 | `print_exchange_frequency()` at t=500 and t=1000 | ✅ Reported |
| Table 9 | Winning actions π̃_i^e(jk\|j) at t=1000 | `print_winning_classifier_actions()` at t=1000 | ✅ Reported |
| Tables 10–15 | Individual classifier strings and strengths for all 3 types | ✅ `print_classifier_strengths()` added for all 3 types | ✅ Reported |
| Figure 6 | Distribution of holdings over time | `plot_holdings_distribution()` | ✅ Reported |

**Qualitative Result:** Paper shows rapid convergence to fundamental equilibrium (Good 1 as medium of exchange). Notebook produces same qualitative convergence. ✅

✅ **Updated:** Classifier strength tables now reported via `print_classifier_strengths()` for all 3 agent types.

### 2.2 Economy A1.2 (Random + GA, Fundamental Equilibrium)

**Parameters:** Match paper (Table 16). ✅

| Paper Table | Content | Notebook | Status |
|---|---|---|---|
| Table 16 | Holdings at t=1000, t=2000 | Reported at t=1000 (manual) and t=2000 (`print_full_analysis`) | ✅ |
| Table 17 | Exchange freq at t=1000, t=2000 | ✅ Reported at t=1000 and t=2000 via `print_full_analysis(t_idx=999)` | ✅ |
| Table 18 | Winning actions at t=1000, t=2000 | ✅ Reported at t=1000 and t=2000 via `print_full_analysis(t_idx=999)` | ✅ |
| Table 19 | Consumption freq at t=2000 | `print_consumption_frequency()` at t=2000 | ✅ |
| Tables 20–27 | Classifier strings/strengths at t=1000, t=2000 | ✅ `print_classifier_strengths()` at t=1000 and t=2000 | ✅ Reported |
| Figure 7 | Holdings distribution over time | `plot_holdings_distribution()` | ✅ |

✅ **Updated:** Exchange frequency and winning actions now reported at both t=1000 and t=2000 via `print_full_analysis(t_idx=999)`.  
✅ **Updated:** Classifier strength tables (Tables 20–27) now reported at t=1000 and t=2000 via `print_classifier_strengths()`.

### 2.3 Economy A2.1 (Complete Enumeration, High Utility)

**Parameters:** Match paper (Table 28). ✅

| Paper Table | Content | Notebook | Status |
|---|---|---|---|
| Holdings at t=500, t=1000 | π_i^h(j) | Reported at t=500 (manual) and t=1000 | ✅ |
| Exchange freq at t=500, t=1000 | π_i^e(jk) | ✅ Reported at t=500 and t=1000 via `print_full_analysis(t_idx=499)` | ✅ |
| Winning actions at t=500, t=1000 | π̃_i^e(jk\|j) | ✅ Reported at t=500 and t=1000 via `print_full_analysis(t_idx=499)` | ✅ |

**Qualitative Result:** The paper finds convergence to *fundamental* equilibrium despite the unique theoretical equilibrium being speculative. This is attributed to overly general consumption classifiers in type I agents. The notebook reports this finding but does not display the consumption classifiers that explain the failure to support speculative equilibrium. The paper notes that "the classifier $c^1_{\#,0}$ was slowly gaining strength" which requires classifier-level inspection not currently available.

### 2.4 Economy A2.2 (Random + GA, High Utility)

**Parameters:** Match paper. ✅

The paper only provides a brief qualitative summary for A2.2 ("refer the reader to the working paper for tables"). The notebook reports holdings, exchange frequency, winning actions, and consumption frequency at t=2000. This level of detail **exceeds** the paper's reported results — which is appropriate for a companion notebook.

### 2.5 Economy B.1 (Alternative Production, Complete Enumeration)

**Parameters:** Match paper (Table in Section 7.3). ✅  
Note: `bid_trade=(0.25, 0.25)` — correctly 10× larger than other economies, matching the paper.

| Paper Table | Content | Notebook | Status |
|---|---|---|---|
| Holdings at t=500, t=1000 | π_i^h(j) | Reported at t=500 (manual) and t=1000 | ✅ |
| Exchange freq at t=500, t=1000 | π_i^e(jk) | ✅ Reported at t=500 and t=1000 via `print_full_analysis(t_idx=499)` | ✅ |
| Winning actions at t=500, t=1000 | π̃_i^e(jk\|j) | ✅ Reported at t=500 and t=1000 via `print_full_analysis(t_idx=499)` | ✅ |
| Consumption freq at t=500 | π_i^c(j\|j) | ✅ Reported at t=500 via `print_full_analysis(t_idx=499)` | ✅ |

**Key Finding in Paper:** Economy B.1 shows speculative equilibrium at t=500 that transitions to fundamental by t=1000. This dynamic transition is the most interesting result for this economy. ✅ **Updated:** The notebook now reports full analysis at both t=500 and t=1000, including exchange frequency, winning actions, and consumption frequency.

### 2.6 Economy B.2 (Alternative Production, Random + GA)

**Parameters:** Match paper (Table in Section 7.3). ✅  
Note: `bid_trade=(0.025, 0.025)` — correctly different from B.1.

| Paper Table | Content | Notebook | Status |
|---|---|---|---|
| Holdings at t=1000, t=2000 | π_i^h(j) | Reported at t=1000 and t=2000 | ✅ |
| Exchange freq at t=1000, t=2000 | π_i^e(jk) | ✅ Reported at t=1000 and t=2000 via `print_full_analysis(t_idx=999)` | ✅ |
| Winning actions at t=1000, t=2000 | π̃_i^e(jk\|j) | ✅ Reported at t=1000 and t=2000 via `print_full_analysis(t_idx=999)` | ✅ |

### 2.7 Economy C (Fiat Money)

**Parameters:** All match paper. ✅  
- Storage costs: ✅ `[9, 14, 29, 0]`
- Utility: ✅ 100
- Classifiers: ✅ E_a=150, C_a=20
- Bids: ✅ bid_trade=(0.025, 0.025), bid_consume=(0.025, 0.25)
- Fiat money: ✅ 48 units injected
- ~~**GA params: ⚠️ `pcross=0.01, pmutation=0.2`** — appears inverted vs. standard values~~ ✅ **Fixed:** Now `pcross=0.6, pmutation=0.01` matching MATLAB `winitial.m`

| Paper Table | Content | Notebook | Status |
|---|---|---|---|
| Holdings at t=750, t=1250 | π_i^h(j) 4-good | `print_fiat_holdings()` at t=750 and t=1250 | ✅ |
| Exchange freq at t=750 | π_i^e(jk) 4-good | `print_fiat_exchange_freq()` at t=750 | ✅ |
| Exchange freq at t=1250 | π_i^e(jk) 4-good | `print_fiat_exchange_freq()` at t=1250 | ✅ |
| Winning actions at t=750, t=1250 | π̃_i^e(jk\|j) 4-good | `print_fiat_winning_actions()` at both | ✅ |
| Figure 9 | Holdings distribution | 3-subplot figure with 4 lines each | ✅ |

**Economy C has the best table coverage** among all economies.

### 2.8 Economy D (Five Goods, Five Types)

**Parameters:** Match paper. ✅

| Paper Table | Content | Notebook | Status |
|---|---|---|---|
| Holdings at t=500 | π_i^h(j) 5×5 | `print_5good_holdings()` at t=500 | ✅ |
| Holdings at t=1750 | π_i^h(j) 5×5 | `print_5good_holdings()` at t=1750 | ✅ |
| Exchange freq at t=500 | π_i^e(jk) 5×5×5 | `print_5good_exchange_freq()` at t=500 | ✅ |
| Exchange freq at t=1750 | π_i^e(jk) 5×5×5 | `print_5good_exchange_freq()` at t=1750 | ✅ |
| Winning actions at t=500 | π̃_i^e(jk\|j) 5×5 | `print_5good_winning_actions()` at t=500 | ✅ |
| Winning actions at t=1750 | π̃_i^e(jk\|j) 5×5 | `print_5good_winning_actions()` at t=1750 | ✅ |
| Figure 10 | Production structure diagram | Implemented in Cell 54 | ✅ |
| Figure 11 | Exchange pattern diagram | Implemented in Cell 54 | ✅ |

~~**Runtime Bug:**~~ ✅ **Fixed.** The `IndexError` in `print_5good_exchange_freq()`, `print_5good_holdings()`, and `print_5good_winning_actions()` for negative `t_idx` values has been resolved. Functions now convert negative indices to valid positive indices.

**Extra content:** The notebook extends the paper by running Economy D for 3000 periods (paper only reports to t=1750) and includes a trading pattern analysis that checks whether agent strategies follow fundamental equilibrium logic. This is a valuable addition.

---

## 3. Summary: Table Coverage

| Economy | Holdings | Exchange Freq | Winning Actions | Consump. Freq | Classifier Strengths | Figures |
|---|---|---|---|---|---|---|
| A1.1 | ✅ t=500,1000 | ✅ t=500,1000 | ✅ t=1000 | ✅ (extra) | ✅ All 3 types | ✅ Fig 6 |
| A1.2 | ✅ t=1000,2000 | ✅ t=1000,2000 | ✅ t=1000,2000 | ✅ t=2000 | ✅ t=1000,2000 | ✅ Fig 7 |
| A2.1 | ✅ t=500,1000 | ✅ t=500,1000 | ✅ t=500,1000 | ✅ (extra) | N/A | ✅ |
| A2.2 | ✅ t=1000,2000 | ✅ t=2000 | ✅ t=2000 | ✅ t=2000 | N/A | ✅ |
| B.1 | ✅ t=500,1000 | ✅ t=500,1000 | ✅ t=500,1000 | ✅ t=500 | N/A | ✅ Fig 8 |
| B.2 | ✅ t=1000,2000 | ✅ t=1000,2000 | ✅ t=1000,2000 | ✅ t=2000 | N/A | ✅ |
| C | ✅ t=750,1250 | ✅ t=750,1250 | ✅ t=750,1250 | Not in paper | N/A | ✅ Fig 9 |
| D | ✅ t=500,1750 | ✅ t=500,1750 | ✅ t=500,1750 | Not in paper | N/A | ✅ Fixed |

**Legend:** ✅ = matches paper, ⚠️ = partial/incomplete, ❌ = missing

---

## 4. Prioritized Recommendations

### Priority 1 — Bug Fixes (Blocking)

1. ~~**Fix Economy D results cell (Cell 51):**~~ ✅ **Done.** Negative index handling added to `print_5good_holdings()`, `print_5good_exchange_freq()`, and `print_5good_winning_actions()`.

### Priority 2 — Algorithm Improvements (High Impact)

2. ~~**Implement specialization operator:**~~ ✅ **Done.** `apply_specialization()` implemented with $f_s(t) = 1/(2\sqrt{t})$ and integrated into all three simulation classes.

3. ~~**Activate diversification:**~~ ✅ **Done.** `apply_diversification()` now called in `get_trade_decision()` and `get_consume_decision()` for all three agent classes.

4. ~~**Investigate Economy C GA parameters:**~~ ✅ **Done.** Confirmed as a bug by reading original MATLAB code (`class003.m` calls `winitial` with no overrides; `winitial.m` sets `pcrosst=0.6, pmutationt=0.01` universally). Fixed to `ga_pcross=0.6, ga_pmutation=0.01`.

5. ~~**Refine GA parameter mapping:**~~ ✅ **Done.** GA rewritten to replicate `ga3.m` from MATLAB: two-stage parent selection (usage-weighted pre-selection p₂=0.7, then fitness-proportional roulette wheel), single-point crossover (not two-point), ternary cyclic mutation, crowding-based replacement via `cankill` set. `EconomyConfig` extended with all GA parameters. N_pairs now computed from `round(propselect * n_classifiers * 0.5)`, giving 7 trade pairs and 1 consume pair for 72/12 classifiers.

### Priority 3 — Missing Outputs (Medium Impact)

6. ~~**Add intermediate-time reporting for A1.2, A2.1, B.1, B.2:**~~ ✅ **Done.** All economies now use `print_full_analysis(t_idx=...)` for both time snapshots. Economy B.1 consumption frequency corrected to t=500.

7. ~~**Add classifier strength tables for A1.1:**~~ ✅ **Done.** `print_classifier_strengths()` function implemented and called for all 3 agent types in Economy A1.1.

8. ~~**Add classifier strength tables for A1.2:**~~ ✅ **Done.** `print_classifier_strengths()` now called for all 3 agent types at both t=1000 and t=2000, replicating paper Tables 20–27.

### Priority 4 — Code Quality (Maintenance)

9. **Unify agent and simulation classes:** Refactor `ClassifierAgent`, `FiatMoneyAgent`, and `FiveGoodAgent` into a single parameterized agent class. Similarly unify the three simulation classes. This eliminates the risk of divergent bug fixes and reduces notebook length by ~40%.

10. **Extend `EconomyConfig` to cover C and D:** Add fields for `n_fiat` (fiat money count), `n_bits` (encoding width), and `ga_params` dictionary. This enables a single summary function that can compare all 8 economies from their configs.

11. ~~**Populate classifier count constraint:**~~ ✅ **Done.** `create_classifier_replacing_weakest()` replaces most redundant (or weakest) classifier when no match found, matching MATLAB `create.m`. Population size is now constant across all agent classes.

### Priority 5 — Enhanced Analysis (Low Priority, High Value)

12. **Add convergence diagnostics:** The paper discusses convergence criteria in Section 5. Implement a convergence metric (e.g., rolling window standard deviation of holdings distribution) and report whether each economy has converged by its final period.

13. ~~**Add Wicksell triangle visualization for all 3-good economies:**~~ ✅ **Done.** Data-driven `plot_wicksell_triangle()` function reads actual exchange counts from simulation history and draws arrows with thickness proportional to exchange frequency. Called for all 8 economies (A1.1, A1.2, A2.1, A2.2, B.1, B.2, C, D) with appropriate layout (triangle for 3-type, pentagon for 5-type).

14. **Add side-by-side comparison with paper values:** For each table output, display the paper's values alongside the notebook's simulation values. This would make discrepancies immediately visible. Since the simulations are stochastic, exact matches are not expected, but qualitative patterns should align.

15. **Explore bid-based vs strength-based auction:** The paper's informal text describes "highest bidder" winning the auction, while the formal equation uses highest strength. For complete enumeration (uniform specificity), these are equivalent. For random classifiers (varying specificity), bid-based selection would favor more specific rules. Consider adding a parameter to toggle between the two modes.

---

## 5. Numerical Reproducibility Note

The notebook uses fixed random seeds (`seed=42` for most economies), which ensures deterministic results within a single execution environment. However, the paper's results come from different random seeds and (presumably) FORTRAN code. **Exact numerical matches with the paper are not expected.** The assessment criterion should be:

1. **Qualitative convergence:** Does the simulated economy converge to the same type of equilibrium (fundamental vs. speculative) as the paper reports?
2. **Pattern consistency:** Do holdings distributions, exchange frequencies, and winning actions show the same structural patterns (which goods are held, which trades occur)?
3. **Dynamic behavior:** Does the economy exhibit the same time-evolution characteristics (e.g., Economy B.1's transition from speculative to fundamental)?

Qualitative convergence appears correct for economies A1.1, A1.2, B.1, B.2, C, and D. Economy A2.1 should converge to fundamental (matching paper), but verification requires the classifier-level analysis described above. Economy A2.2 is inherently ambiguous (paper reports inconclusive results).

---

## Appendix: Paper-to-Notebook Table Number Mapping

The paper uses continuous numbering for all tables and figures. Below is the mapping:

| Paper Table | Content | Economy | Notebook Location |
|---|---|---|---|
| Table 1 | Encoding scheme | All | Cell 8 (implicit) |
| Table 2 | Summary of economies | All | Cell 57 (summary) |
| Table 3 | A1.1 parameters | A1.1 | Cell 19 |
| Tables 4–6 | Equilibrium probabilities | A1 theory | Cell 18 (markdown) |
| Table 7 | A1.1 holdings | A1.1 | Cell 21 |
| Table 8 | A1.1 exchange freq | A1.1 | Cell 21 |
| Table 9 | A1.1 winning actions | A1.1 | Cell 21 |
| Tables 10–15 | A1.1 classifier strengths | A1.1 | ✅ `print_classifier_strengths()` |
| Figure 6 | A1.1 holdings plot | A1.1 | Cell 21–22 |
| Table 16 | A1.2 parameters | A1.2 | Cell 25 |
| Tables 17–19 | A1.2 holdings/exchange/winning | A1.2 | Cell 26 |
| Table 20–27 | A1.2 classifier strengths | A1.2 | ✅ `print_classifier_strengths()` at t=1000 and t=2000 |
| Figure 7 | A1.2 holdings plot | A1.2 | Cell 26 |
| Table 28 | A2.1 parameters | A2.1 | Cell 29 |
| A2 equilibrium tables | Speculative equilibrium probs | A2 theory | Cell 28 (markdown) |
| A2.1 results | Holdings/exchange/winning | A2.1 | Cell 30 |
| B.1 parameters | — | B.1 | Cell 37 |
| B equilibrium tables | Fundamental/speculative probs | B theory | Cell 36 (markdown) |
| B.1 results | Holdings/exchange/winning/consump | B.1 | Cell 38 |
| B.2 parameters/results | — | B.2 | Cells 40–41 |
| C parameters | — | C | Cell 44 |
| C equilibrium tables | Fundamental equilibrium | C theory | Cell 43 (markdown) |
| C results | Holdings/exchange/winning | C | Cell 46 |
| Figure 9 | C holdings plot | C | Cell 46 |
| D parameters | — | D | Cell 49 |
| D results | Holdings/exchange/winning | D | Cells 51–54 |
| Figures 10–11 | D production/exchange patterns | D | Cell 54 |
