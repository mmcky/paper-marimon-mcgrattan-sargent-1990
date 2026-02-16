# Copilot Instructions

## Project Overview

This repository contains a replication of "Money as a Medium of Exchange in an Economy with Artificially Intelligent Agents" by Marimon, McGrattan, and Sargent (1990), published in the Journal of Economic Dynamics and Control, 14, 329–373.

The companion notebooks in `jupyter/` implement the Kiyotaki-Wright model with Holland classifier systems in Python (numpy, matplotlib).

## Shell Scripting Rules

**Never pipe heredocs directly into interpreters in zsh** (e.g., `python3 << 'EOF'`). These break with special characters, quotes, and nested escaping.

Instead, write multi-line content to a temporary file, execute it, and clean up:

```bash
# BAD — piping heredoc directly into interpreter
python3 << 'EOF'
import json
# ... code with quotes, special chars ...
EOF

# GOOD — write to temp file, execute, clean up
cat > .tmp/script.py << 'EOF'
import json
# ... code with quotes, special chars ...
EOF
python3 .tmp/script.py
rm .tmp/script.py
```

Using heredocs to write files (via `cat >`) is fine — the problem is piping them directly into commands. The `.tmp/` directory is gitignored and exists for this purpose.

## Notebook Editing

- The `edit_notebook_file` tool edits the VS Code buffer, not the disk file directly
- Terminal commands edit the disk file, not the VS Code buffer
- When both are used, the user must save the VS Code buffer to sync changes to disk
- After terminal-based edits to `.ipynb` files, the notebook may need to be reloaded in VS Code

## Key Files

| Path | Description |
|------|-------------|
| `jupyter/companion-notebook-1.ipynb` | Main replication notebook (8 economies: A1.1, A1.2, A2.1, A2.2, B.1, B.2, C, D) |
| `jupyter/companion-notebook-2-alphago.ipynb` | AlphaGo-style approach |
| `paper/paper.md` | MyST Markdown version of the paper |
| `paper/references.bib` | Bibliography |
| `original/Marimon_McGrattan_Sargent_1990.pdf` | Original paper PDF |
| `original/matlab/` | Original MATLAB code (McGrattan, 1989) |
| `src/` | Python port of original MATLAB (work in progress) |

## Paper Reference

Key equations to match:
- **Eq (10)**: Consumption classifier strength update — denominator is τ_c − 1 (n_used BEFORE incrementing)
- **Eq (11)**: Exchange classifier strength update — denominator is τ_e (n_used BEFORE incrementing)  
- **Eq (12)**: Utility function — u_i(k) = 0 for k ≠ i (agents can consume wrong goods but get zero utility)

8 economies with specific parameter sets defined in Tables 1–22 of the paper.
