# Traceability Matrix

This document maps the figures and tables presented in the manuscript to their corresponding generating scripts and output files in the repository.

| Manuscript Element | Generating Script | Output Directory | Notes |
| :--- | :--- | :--- | :--- |
| **Table 14** (Intrinsic Dimensionality) | `experiments/intrinsic_dimensionality.py` | `results/experiments/dimensionality` | Reports TwoNN dimensionality |
| **Table 15** (Main Confirmatory Comparison) | `experiments/four_way_ablation.py` | `results/experiments/ablation` | Hypervolume Wilcoxon tests |
| **Table 16** (Decoupling Attribution) | `experiments/decoupling.py` | `results/experiments/ablation` | Isolates initialization vs replacement |
| **Table 17** (Anytime Convergence) | `experiments/anytime_analysis.py` | `results/experiments/anytime` | Performance at fixed budget fractions |
| **Table 18** (Normalization Sensitivity) | `experiments/normalization_schemes.py` | `results/experiments/normalization` | Static vs Dynamic normalization |
| **Table 19** (Population Equalization) | `experiments/popsize_equalization.py` | `results/experiments/popsize` | Controls for $N$ advantage |
| **Table 20** (Rho Sensitivity) | `experiments/rho_sweep.py` | `results/experiments/rho` | Sensitivity to $\rho$ scaling |
| **Figure 17** (Hyperparameter Grid) | `experiments/beta_phi_grid.py` | `results/experiments/grid` | Stabilization $\beta, \phi$ search |

To regenerate all results automatically in the correct order, run `./run_campaign.sh`.
