# Manuscript Traceability

This document provides the mapping between the elements of the article and the code/data artifacts in this repository.

| Manuscript Element | Script | Input Data | Output Artifact | Expected Status |
|-------------------|--------|------------|-----------------|-----------------|
| Table 2 (Predictive Validity) | `src/comfort_models.py` | `04_comfort_scenario_ratings.csv` | `comfort_model_metrics.json` | Reproducible |
| Table 3 (Priority weights) | `src/preferences/stabilization.py` | `05_pairwise_objective_preferences.csv` | `table3_stabilized_weights.csv` | Reproducible |
| Table 5 (Strata Distribution) | `src/survey_data_loader.py` | `01_demographics_profiles.csv` | - | Automated load |
| Table 6 (Network descriptors) | `src/network/descriptors.py` | `strasbourg_multimodal.graphml` | `table6_network_descriptors.csv` | Reproducible |
| Table 8 (Benchmark statistics) | `reproduce.sh --all` | - | `table8_main_statistics.csv` | Reproducible |
| Tables 9-12 (Ranks & Correlations) | `reproduce.sh --all` | - | `table9_12_*.csv` | Reproducible |
| Tables 13-17 (Sensitivity & Ablation) | `reproduce.sh --all` | - | `table13_17_*.csv` | Reproducible |
| Figures 3-9 (Plots) | `reproduce.sh --all` | - | `plots/*.pdf` | Reproducible |
