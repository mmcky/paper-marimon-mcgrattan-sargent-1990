# Changelog

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
