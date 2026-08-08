# Manuscript Traceability

This document provides the mapping between the elements of the article and the code/data artifacts in this repository.

| Manuscript Element | Script | Input Data | Output Artifact | Expected Status |
|-------------------|--------|------------|-----------------|-----------------|
| Table 2 (Profiles) | `src/survey_data_loader.py` | `01_demographics_profiles.csv` | - | Automated load |
| Table 3 (Priority weights) | `src/preferences/stabilization.py` | `05_pairwise_objective_preferences.csv` | `table3_stabilized_weights.csv` | Reproducible |
| Table 5 (Hold-out metrics) | `src/comfort_models.py` | `04_comfort_scenario_ratings.csv` | `comfort_model_metrics.json` | Reproducible |
| Table 6 (Network descriptors) | `src/network/descriptors.py` | `strasbourg_multimodal.graphml` | `table6_network_descriptors.csv` | Reproducible |
| Table 8 (Benchmark statistics) | `experiments/run_campaign.sh` | - | `table8_main_statistics.csv` | Pending campaign |
| Tables 9-12 (Ranks & Correlations) | `experiments/run_campaign.sh` | - | `table9_12_*.csv` | Pending campaign |
| Tables 13-17 (Sensitivity & Ablation) | `experiments/run_campaign.sh` | - | `table13_17_*.csv` | Pending campaign |
| Figures 3-9 (Plots) | `experiments/run_campaign.sh` | - | `plots/*.pdf` | Pending campaign |

*Note: As noted in the main `README.md`, all numerical results are currently being regenerated to reflect the finalized discrete path-based formulation.*
