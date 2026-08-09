#!/bin/bash
set -e

echo "Starting full reproduction campaign..."

# Data prep and network
echo "[1/4] Preparing data and network..."
python -m src.network.builder
python -m src.network.descriptors --out results/network

# Main campaign
echo "[2/4] Running main optimization campaign (This will take a long time)..."
python -m src.pipeline_V6_smart --output-dir results/outputs_main --plans main extended convergence ablation --max-workers 3

# Sensitivities
echo "[3/4] Running sensitivities..."
python -m experiments.four_way_ablation
python -m experiments.decoupling
python -m experiments.anytime_analysis
python -m experiments.intrinsic_dimensionality
python -m experiments.normalization_schemes
python -m experiments.popsize_equalization
python -m experiments.rho_sweep
python -m experiments.beta_phi_grid

# Analytics
echo "[4/4] Generating analytics..."
python -m src.comfort_models
python -m src.preferences.stabilization
python -m src.analytics_V6 --runs results/outputs_main --out results/analytics

echo "Reproduction complete."
