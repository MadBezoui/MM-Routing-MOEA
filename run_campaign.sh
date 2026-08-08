#!/bin/bash
set -e

echo "Starting full experimental campaign for MM-Routing-MOEA..."
python -m src.pipeline_V6_smart --output-dir results/outputs_main \
    --plans main extended convergence ablation --max-workers 3

echo "Running analytics to generate tables and plots..."
python -m src.analytics_V6 --runs results/outputs_main --out results/analytics

echo "Campaign complete!"
