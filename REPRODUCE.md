# Exact Reproducibility Instructions

This document maps the manuscript's results directly to the scripts and commands required to generate them.

## Prerequisites
Ensure dependencies match the pinned environment:
```bash
pip install -r requirements.txt
```

## Quick Execution
To regenerate all tables and figures automatically:
```bash
make reproduce
```
Alternatively, use the shell script directly:
```bash
./reproduce.sh --all
```

## Step-by-Step Mapping

### 1. Data Preparation and Network
* **Table 6 (Network descriptors)**
  ```bash
  python -m src.network.builder
  python -m src.network.descriptors --out results/network
  ```
* **Table 5 (Strata Distribution)**
  Generated automatically during data loading in `src/survey_data_loader.py`.

### 2. Main Optimization Campaign
The main campaign runs the optimization plans:
```bash
python -m src.pipeline_V6_smart --output-dir results/outputs_main --plans main extended convergence ablation --max-workers 3
```

### 3. Analytics and Main Tables
* **Table 8, 9, 10, 11, 12 and Figures 3-9**
  ```bash
  python -m src.analytics_V6 --runs results/outputs_main --out results/analytics
  ```

### 4. Sensitivities and Additional Scripts
* **Table 2 (Predictive Validity)**
  ```bash
  python -m src.comfort_models
  ```
* **Table 3 (Priority weights)**
  ```bash
  python -m src.preferences.stabilization
  ```
* **Tables 13-17 (Sensitivity & Ablation)**
  ```bash
  python -m experiments.four_way_ablation
  python -m experiments.decoupling
  python -m experiments.anytime_analysis
  python -m experiments.intrinsic_dimensionality
  python -m experiments.normalization_schemes
  python -m experiments.popsize_equalization
  python -m experiments.rho_sweep
  python -m experiments.beta_phi_grid
  ```
