# MM-Routing-MOEA Reproducibility Package

This repository contains the source code, optimization framework, and analytics scripts for the manuscript on preference-informed multimodal routing.

## Requirements

Execution requires Python 3.11+. The environment is pinned in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Running the Experiments

To guarantee reproducible, deterministic execution of the full experimental campaign (including the generation of all manuscript figures and tables), run the master shell script:

```bash
./run_campaign.sh
```

This will automatically execute the optimization sequences using a parallel `ProcessPoolExecutor` with explicitly controlled RNG seeding. Deterministic under the pinned software environment and tested execution modes. Cross-platform bitwise identity is not guaranteed; statistical reproducibility is expected.

**Note:** The full campaign involves extensive multi-objective optimization across 150 routing profiles and multiple stochastic seeds. Full execution may take several hours depending on your hardware.

## Contents of the Campaign

1. **Smoke Test (`smoke_test.py`)**: A rapid end-to-end verification of the pipeline.
2. **Four-Way Ablation & Decoupling (`four_way_ablation.py`, `decoupling.py`)**: Confirmatory results and decoupling attribution (Tables 15, 16).
3. **Anytime Analysis (`anytime_analysis.py`)**: Convergence speed across normalized budget fractions (Table 17).
4. **Intrinsic Dimensionality (`intrinsic_dimensionality.py`)**: Objective space complexity via TwoNN (Table 14).
5. **Normalization Schemes (`normalization_schemes.py`)**: Dynamic vs. static nadir sensitivity (Table 18).
6. **Population Size Equalization (`popsize_equalization.py`)**: Sensitivity to exact matching of population sizes across algorithms (Table 19).
7. **Rho Sensitivity (`rho_sweep.py`)**: Reference point scaling robustness (Table 20).
8. **Beta/Phi Grid (`beta_phi_grid.py`)**: Stabilization hyperparameter landscape (Figure 17).

All outputs are saved to `results/experiments/`.
