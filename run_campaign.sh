#!/usr/bin/env bash
# run_campaign.sh
# 
# Master execution script for the MM-Routing-MOEA experimental campaign.
# This script executes all necessary optimization runs and analytics generation
# in the correct dependency order.
#
# Requirements:
# - Python 3.11+
# - Dependencies installed via: pip install -r requirements.txt

set -e

echo "========================================================="
echo " MM-Routing-MOEA: Definitive Experimental Campaign"
echo "========================================================="

# Create results directory
mkdir -p results/experiments

echo "[1/8] Running Smoke Test to verify environment..."
python -m experiments.smoke_test --out results/experiments/smoke

echo "[2/8] Executing Main Decoupling Ablation (Table 15, 16)..."
# This performs the 4-way ablation: PI-NSGA-III (stab), PI-NSGA-III (raw), NSGA-III, NSGA-II
python -m experiments.four_way_ablation --out results/experiments/ablation
python -m experiments.decoupling --out results/experiments/ablation

echo "[3/8] Executing Anytime Analysis (Table 17)..."
# Generates convergence budget fractions vs terminal performance
python -m experiments.anytime_analysis --out results/experiments/anytime

echo "[4/8] Executing Intrinsic Dimensionality Analysis (Table 14)..."
# Evaluates the TwoNN dimension estimator on the objective space
python -m experiments.intrinsic_dimensionality --out results/experiments/dimensionality

echo "[5/8] Executing Normalization Schemes Sensitivity (Table 18)..."
# Evaluates dynamic vs static normalization
python -m experiments.normalization_schemes --out results/experiments/normalization

echo "[6/8] Executing Population Size Equalization (Table 19)..."
# Evaluates the effect of exact population size matching
python -m experiments.popsize_equalization --out results/experiments/popsize

echo "[7/8] Executing Rho Sensitivity Analysis (Table 20)..."
# Tests reference point scaling factor rho
python -m experiments.rho_sweep --out results/experiments/rho

echo "[8/8] Executing Beta/Phi Grid Search (Figure 17)..."
# Grid search over stabilization hyperparameters
python -m experiments.beta_phi_grid --out results/experiments/grid

echo "========================================================="
echo "Campaign successfully completed. All results are stored in results/experiments/"
echo "========================================================="
