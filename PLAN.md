# PLAN: MATLAB to Python & Paper to MyST Conversion

This project converts 1989 research code by Ellen R. McGrattan implementing **Genetic Algorithms** and **Classifier Systems** for economic simulations (Wicksell's N-tangles model studying commodity money emergence).

**Goal:** Create a reproducible scientific paper with modern Python code and MyST Markdown documentation, renderable with `mystmd`.

**Approach:** Use local Python modules (no packaging) with NumPy for scientific computing.

---

## Project Structure

```
matlab-project/
├── original/                  # Original files (preserved)
│   ├── index.html
│   ├── robots.txt
│   └── data/mms/           # Original MATLAB code
├── src/                     # New Python modules
├── docs/                    # MyST documentation
├── paper/                   # Converted paper
├── PLAN.md
├── requirements.txt
└── myst.yml
```

---

## Phase 1: Code Conversion & Documentation

Convert all MATLAB code to Python and documentation to MyST Markdown.

### 1.1 Project Setup

- [x] Create `src/` directory for Python modules
- [x] Create `docs/` directory for MyST documentation
- [x] Create `requirements.txt` with dependencies (numpy, matplotlib)
- [x] Create `myst.yml` configuration for MyST rendering
- [x] Create conda environment `mms-ga` with Python 3.11

### 1.2 Utility Modules (Low Complexity)

Source: `original/data/mms/`

- [x] `decode.m` → `src/decode.py` — binary string to real number conversion
- [x] `select.m` → `src/selection.py` — roulette wheel selection (renamed to avoid Python stdlib conflict)
- [x] `statistics.m` → `src/statistics.py` — compute max, min, avg, sum fitness
- [x] `objfunc.m` → `src/objfunc.py` — objective function (Rosenbrock)
- [x] `scalepop.m` → `src/scaling.py` — linear scaling for negative fitness
- [x] `scalestr.m` → `src/scaling.py` — strength scaling for classifiers

### 1.3 Crowding Modules (Low Complexity)

Source: `original/data/mms/`

- [x] `crowding.m` → `src/crowding.py` — find similar weak classifier to replace
- [x] `crowding2.m` → `src/crowding.py` — variant with different strength position
- [x] `crowding3.m` → `src/crowding.py` — advanced crowding
- [x] `crowdin3.m` → `src/crowding.py` — alternate implementation

### 1.4 Genetic Algorithm Modules (Medium Complexity)

Source: `original/data/mms/`

- [x] `ga.m` → `src/ga.py` — base GA for classifier systems, single-point crossover
- [x] `ga2.m` → `src/ga.py` — GA with proportional-used selection weighting
- [x] `ga3.m` → `src/ga.py` — GA with kill-eligibility constraints (uratio)
- [x] `ga4.m` → `src/ga.py` — GA with 2-point crossover
- [x] `create.m` → `src/create.py` — create new classifier for unmatched condition
- [x] `cre.m` → `src/create.py` — alternate create implementation

### 1.5 Configuration Modules (Low-Medium Complexity)

Source: `original/data/mms/`

- [x] `initdata.m` → `src/config.py` — SGA parameter configuration
- [x] `winitial.m` → `src/config.py` — Wicksell parameter configuration
- [x] `wtinit.m` → `src/config.py` — simple Wicksell parameters

### 1.6 Main Programs (High Complexity)

Source: `original/data/mms/`

- [x] `sga.m` → `src/sga.py` — Simple Genetic Algorithm optimizer (~400 lines)
- [x] `resume.m` → `src/sga.py` — resume interrupted SGA run

### 1.7 Classifier Simulations (Very High Complexity)

Source: `original/data/mms/`

- [x] `wicksell.m` → `src/classifier_simulation.py` — incorporated into ClassifierSimulation
- [x] `wnew.m` → `src/classifier_simulation.py` — incorporated into ClassifierSimulation
- [x] `class001.m`–`class004.m` → `src/classifier_simulation.py` — consolidated `ClassifierSimulation` class
- [x] Create `src/experiments/experiment_001.py` — config matching `class001.m`
- [x] Create `src/experiments/experiment_002.py` — config matching `class002.m`
- [x] Create `src/experiments/experiment_003.py` — config matching `class003.m`
- [x] Create `src/experiments/experiment_004.py` — config matching `class004.m`

### 1.8 Documentation Conversion

- [x] `original/data/mms/Readme` → `docs/readme.md` — convert to MyST format
- [x] `original/data/mms/gdisc.tex` → `docs/algorithm.md` — GA algorithm description in MyST
- [x] `original/index.html` → `docs/index.md` — project overview in MyST

### 1.9 Unit Tests

- [x] Create `tests/test_utilities.py` — tests for utility modules
- [x] Create `tests/test_config.py` — tests for configuration classes

---

## Phase 2: Paper Conversion

Convert the published paper to MyST Markdown for reproducible rendering.

### 2.1 Paper Structure

- [x] Extract text content from `Marimon_McGrattan_Sargent_1990.pdf`
- [x] Create `paper/paper.md` with MyST frontmatter
- [x] Convert sections to MyST Markdown headings

### 2.2 Mathematical Content

- [x] Convert equations to LaTeX math blocks (`$$...$$`)
- [x] Convert inline math to `$...$`
- [ ] Verify equation rendering with mystmd

### 2.3 Figures and Tables

- [ ] Extract or recreate figures from PDF
- [x] Convert tables to MyST/Markdown table format
- [ ] Add figure captions and cross-references

### 2.4 References

- [x] Create `paper/references.bib` with bibliography
- [x] Add citation keys using MyST syntax `{cite}`
- [x] Configure bibliography rendering in myst.yml

### 2.5 Integration

- [ ] Link paper to code via MyST cross-references
- [ ] Add code blocks showing Python equivalents where relevant
- [x] Test full build with `mystmd build`

---

## Technical Notes

### MATLAB to Python Translation Patterns

| MATLAB | Python/NumPy |
|--------|--------------|
| `ones(n,m)` | `np.ones((n,m))` |
| `rand(n,m)` | `np.random.rand(n,m)` |
| `A'` (transpose) | `A.T` |
| `A.*B` | `A * B` |
| `find(A)` | `np.where(A)` |
| `ceil(rand*n)` | `np.random.randint(1, n+1)` |
| 1-indexed | 0-indexed |
| `eval(['var',num])` | `vars[f"var{num}"]` (dict) |

### Key Design Decisions

1. **No packaging** — use local imports for scientific paper context
2. **Dictionary-based** — replace MATLAB `eval()` with Python dicts
3. **Class-based agents** — `AgentType` class for `CS1`, `CS2`, etc.
4. **NumPy throughout** — vectorized operations for performance

---

## Open Questions

- [x] ~~Consolidate class001–004 variants or keep separate?~~ → **Hybrid approach**: consolidated `ClassifierSimulation` class + separate experiment configs
- [x] ~~Include validation tests against MATLAB outputs?~~ → **Unit tests for utility functions** now; MATLAB output validation can be added later if outputs become available
- [x] ~~Target mystmd CLI or Jupyter Book for rendering?~~ → **mystmd CLI** for lightweight paper rendering
