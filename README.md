# Money as a Medium of Exchange in an Economy with Artificially Intelligent Agents

**Marimon, McGrattan, and Sargent (1990)**

## About

This repository contains:

1. A **MyST Markdown version** of the paper, converted from the [original PDF](original/Marimon_McGrattan_Sargent_1990.pdf)
2. A **Python replication** of all 8 classifier system economies from the paper
3. An **experimental AlphaGo-style approach** to the same problem

> Marimon, R., McGrattan, E., & Sargent, T. J. (1990). Money as a Medium of Exchange in an Economy with Artificially Intelligent Agents. *Journal of Economic Dynamics and Control*, 14(2), 329-373. [DOI: 10.1016/0165-1889(90)90025-C](https://doi.org/10.1016/0165-1889(90)90025-C)

## Project Structure

```
├── paper/                      # MyST paper source
│   ├── paper.md                # Full paper in MyST Markdown
│   ├── COMPARISON_REPORT.md    # PDF → MyST conversion quality report
│   ├── figures/                # Paper figures
│   └── references.bib          # Bibliography
├── jupyter/                    # Companion Jupyter notebooks
│   ├── companion-notebook-1.ipynb   # Python replication (all 8 economies)
│   ├── companion-notebook-2-alphago.ipynb  # AlphaGo experiment
│   └── tom/                    # Alternative implementations (reference)
├── website/                    # Site pages and reports
│   ├── index.md                # Landing page
│   ├── replication/            # Replication reports
│   │   ├── quality-assessment.md
│   │   ├── deepseek-comparison.md
│   │   └── changelog.md
│   └── experiments/            # Experiment reports
│       └── alphago-comparison.md
├── original/                   # Original source materials
│   ├── Marimon_McGrattan_Sargent_1990.pdf  # Original paper PDF
│   └── matlab/                 # Original MATLAB code (McGrattan, 1989)
├── _archive/                   # Construction & intermediate files
│   ├── tests/                  # Test suite for src/
│   ├── scripts/                # Utility scripts
│   └── ...                     # Conversion artifacts
├── src/                        # Python port of original MATLAB (work in progress)
│   ├── README.md               # Module documentation
│   ├── classifier_simulation.py
│   ├── ga.py
│   └── ...
├── myst.yml                    # MyST site configuration
└── requirements.txt            # Python dependencies
```

## Building the Companion Site

```bash
# Install mystmd
npm install -g mystmd

# Start development server
myst start

# Build HTML
myst build --html

# Build PDF of the paper
myst build --pdf
```

## Running the Notebooks

```bash
pip install -r requirements.txt
jupyter lab jupyter/companion-notebook-1.ipynb
```

## Running Tests

```bash
cd reference && pytest tests/ -v
```

## License

This is an academic reproduction for research and educational purposes.
