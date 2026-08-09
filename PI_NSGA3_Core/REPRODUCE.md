# REPRODUCE.md

This file maps every figure and table in the manuscript to the script responsible for generating it, along with the exact output file and the command to run.

> **Prerequisites** — install dependencies once:
> ```bash
> pip install numpy pandas scipy matplotlib scikit-learn pymoo
> ```
> All commands below assume you are running from the **`PI_NSGA3_Core/`** root.

---

## Step 0 — Run the full pipeline (generates raw results)

> Skip this step if `results/outputs_v5_parallel_3threads_fixed/` already exists.

```bash
python src/pipeline_V6_smart.py \
    --survey_dir   data/survey_results \
    --output_dir   results/outputs_v5_parallel_3threads_fixed \
    --max_workers  5
```

**Outputs:** `results/outputs_v5_parallel_3threads_fixed/` — all population checkpoints, recovered HV/IGD CSVs, and comfort-model artefacts.

---

## Step 1 — Generate analytics figures and tables

```bash
python src/analytics_V6.py \
    --output_dir   results/outputs_v5_parallel_3threads_fixed \
    --analytics_dir results/outputs_v5_parallel_3threads_fixed/analytics
```

---

## Figures

| Figure in manuscript | Output file | Generating function | Script |
|---|---|---|---|
| Fig. 1 — Normalized HV distribution (boxplot, all plans) | `analytics/fig_box_normalized_hv.png` | `plot_box_normalized_hv()` | `src/analytics_V6.py` |
| Fig. 2 — Normalized HV violin plot per plan | `analytics/fig_violin_per_plan.png` | `plot_violin_per_plan()` | `src/analytics_V6.py` |
| Fig. 3 — ECDF of normalized HV | `analytics/fig_ecdf_normalized_hv.png` | `plot_ecdf()` | `src/analytics_V6.py` |
| Fig. 4 — HV per-plan strip | `analytics/fig_strip_per_plan.png` | `plot_strip_per_plan()` | `src/analytics_V6.py` |
| Fig. 5 — Hypervolume convergence (analytics, 10-profile plan) | `analytics/fig_hv_convergence.png` | `plot_history_metric()` | `src/analytics_V6.py` |
| Fig. 6 — Per-profile HV convergence grid | `analytics/fig_hv_convergence_per_profile.png` | `plot_hv_per_profile_grid()` | `src/analytics_V6.py` |
| Fig. 9 — NSGA-II vs NSGA-III scatter (normalized HV) | `analytics/fig_scatter_nsga2_vs_nsga3.png` | `plot_scatter_pair()` | `src/analytics_V6.py` |
| Fig. 10 — Caterpillar plot (NSGA-III − NSGA-II Δ per profile) | `analytics/fig_caterpillar_nsga3_minus_nsga2.png` | `plot_caterpillar()` | `src/analytics_V6.py` |
| Fig. 11 — Scatter matrix (all algorithm pairs) | `analytics/fig_scatter_matrix_all_pairs.png` | `plot_scatter_matrix_all_pairs()` | `src/analytics_V6.py` |
| Fig. 12 — Δz distribution (per-profile HV advantage) | `analytics/fig_dz_distribution.png` | `plot_dz_distribution()` | `src/analytics_V6.py` |
| Fig. 13 — Boxplot by commuter archetype | `analytics/fig_box_per_archetype.png` | `plot_box_per_archetype()` | `src/analytics_V6.py` |
| Fig. 14 — Boxplot by trip distance bin | `analytics/fig_box_per_trip_distance.png` | `plot_box_per_trip_distance()` | `src/analytics_V6.py` |
| Fig. 15 — Bar chart by archetype | `analytics/fig_bar_per_archetype.png` | `plot_bar_per_strata()` | `src/analytics_V6.py` |
| Fig. 16 — Bar chart by trip distance | `analytics/fig_bar_per_trip_distance.png` | `plot_bar_per_strata()` | `src/analytics_V6.py` |
| Fig. 17 — Strata heatmap (profile × algorithm coverage) | `analytics/fig_heatmap_strata.png` | `plot_heatmap_strata()` | `src/analytics_V6.py` |
| Fig. 18 — Critical Difference (CD) diagram | `analytics/fig_cd_diagram.png` | `plot_cd_diagram()` | `src/analytics_V6.py` |
| Fig. 19 — Spacing metric convergence | `analytics/fig_spacing_convergence.png` | `plot_history_metric()` | `src/analytics_V6.py` |
| Fig. 20 — Feasibility ratio convergence | `analytics/fig_feasible_ratio_convergence.png` | `plot_history_metric()` | `src/analytics_V6.py` |
| Fig. 21 — Pareto fronts (2D projections) | `analytics/fig_pareto_fronts_2d.png` | `plot_pareto_fronts_2d()` | `src/analytics_V6.py` |
| Fig. 22 — Pareto front (3D) | `analytics/fig_pareto_3d.png` | `plot_pareto_3d()` | `src/analytics_V6.py` |
| Fig. 23 — PCA of solution space | `analytics/fig_pca_solutions.png` | `plot_pca_solutions()` | `src/analytics_V6.py` |
| Fig. 24 — Objective correlation heatmap | `analytics/fig_objective_correlations.png` | `plot_objective_correlations()` | `src/analytics_V6.py` |
| Fig. 25 — Parallel coordinates (objective space) | `analytics/fig_parallel_coordinates.png` | `plot_parallel_coordinates()` | `src/analytics_V6.py` |
| Fig. 26 — Mode share per algorithm | `analytics/fig_mode_share_per_algorithm.png` | `plot_mode_share_per_algorithm()` | `src/analytics_V6.py` |
| Fig. 27 — Runtime comparison | `analytics/fig_runtime_comparison.png` | `plot_runtime_comparison()` | `src/analytics_V6.py` |
| Fig. 28 — Comfort model comparison (predictions vs. ground truth) | `analytics/fig_comfort_model_comparison.png` | `plot_comfort_models()` | `src/analytics_V6.py` |
| Fig. 29 — Comfort model regional errors | `analytics/fig_comfort_region_errors.png` | `plot_comfort_region_errors()` | `src/analytics_V6.py` |

---

## Tables

| Table in manuscript | Output file | Description | Script |
|---|---|---|---|
| Table 1 — Mean / Median / Std of normalized HV per plan | `analytics/summary_stats_per_plan.csv` | Core performance summary across all plans | `src/analytics_V6.py` |
| Table 2 — Paired effect sizes (Cliff's Δ / Cohen's d) | `analytics/paired_effect_sizes_per_plan.csv` | Pairwise comparison NSGA-II vs NSGA-III | `src/analytics_V6.py` |
| Table 3 — Win / Tie / Loss counts per plan | `analytics/win_tie_loss_per_plan.csv` | Head-to-head algorithm win rates per profile | `src/analytics_V6.py` |
| Table 4 — Friedman test pivot (extended benchmark) | `analytics/friedman_pivot_extended.csv` | Non-parametric multi-algorithm ranking | `src/analytics_V6.py` |
| Table 5 — Per-archetype HV summary | `analytics/per_archetype_summary.csv` | Breakdown by commuter archetype | `src/analytics_V6.py` |
| Table 6 — Per-trip-distance HV summary | `analytics/per_trip_distance_summary.csv` | Breakdown by trip distance bin | `src/analytics_V6.py` |
| Table 7 — Strata breakdown per plan | `analytics/strata_breakdown_per_plan.csv` | Profile coverage per archetype × distance cell | `src/analytics_V6.py` |
| Table 8 — Objective correlations | `analytics/objective_correlations.csv` | Spearman correlations between objectives | `src/analytics_V6.py` |
| Table 9 — Per-algorithm runtime | `analytics/per_algorithm_runtime_summary.csv` | Wall-clock time statistics | `src/analytics_V6.py` |
| Table 10 — Comfort model metrics | `results/outputs_v5_parallel_3threads_fixed/comfort_model_comparison.csv` | MAE / R² for heuristic, linear, and MLP models | `src/pipeline_V6_smart.py` |
| Table 11 — Objective weights (raw vs. stabilized) | `results/outputs_v5_parallel_3threads_fixed/objective_weights_raw.json` / `objective_weights_stabilized.json` | Survey-derived vs. Hamilton-stabilized weights | `src/pipeline_V6_smart.py` |
| Table 12 — Sampling audit (profiles per plan) | `results/outputs_v5_parallel_3threads_fixed/*_sampling_audit.csv` | Strata allocation per run plan | `src/pipeline_V6_smart.py` |

## Complete Reproduction (all figures + tables in one pass)

```bash
# 1. Run the main pipeline (skip if results already exist)
python src/pipeline_V6_smart.py \
    --survey_dir data/survey_results \
    --output_dir results/outputs_v5_parallel_3threads_fixed \
    --max_workers 5

# 2. Generate all analytics figures and tables
python src/analytics_V6.py \
    --output_dir   results/outputs_v5_parallel_3threads_fixed \
    --analytics_dir results/outputs_v5_parallel_3threads_fixed/analytics
```

---

## Data Flow Summary

```
data/survey_results/
        │
        ▼
src/pipeline_V6_smart.py   ──────────────────────────► results/outputs_v5_parallel_3threads_fixed/
                                                              │
                                (comfort models,              │
                                 HV/IGD recovery,             ▼
                                 weight audit)       src/analytics_V6.py
                                                              │
                                                   ┌──────────┴──────────┐
                                                   ▼                     ▼
                                        analytics/ figures          analytics/ tables
                                        (fig_*.png)                 (*_summary.csv)
```
