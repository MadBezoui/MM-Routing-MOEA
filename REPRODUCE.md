# REPRODUCE.md

Maps every numbered table and figure of the manuscript to the command that
produces it. Manuscript numbering, not internal numbering.

> **Install once**
> ```bash
> pip install numpy pandas scipy matplotlib scikit-learn pymoo networkx tqdm
> ```
> Run every command from the repository root. All modules are invoked with
> `python -m`, so no `PYTHONPATH` setting is needed.

---

## Step 0 — Build the multimodal network

Required once. Streams the frozen OSM extract and the archived GTFS feed into
the consolidated five-mode graph of Section 5.2.

```bash
# fetch the frozen inputs (see README for the snapshot identifiers)
docker build -f Dockerfile.data -t pi-nsga3-data . && docker run --rm -v "${PWD}:/workspace" pi-nsga3-data

python -m src.network.builder
python -m src.network.descriptors --out results/network
```

**Produces** `data/processed/strasbourg_multimodal.graphml`, plus
`results/network/` containing `table6_network_descriptors.csv`,
`network_descriptors.json`, the five per-mode adjacency matrices and the
node-degree histogram.

The first run parses the 363 MB OSM XML (about 15 s) and caches the road layers
to `data/processed/osm_road_layers.npz`; later runs reuse the cache.

---

## Step 1 — Priority weights

```bash
python -m src.preferences.stabilization --out results/preferences
```

**Produces** `objective_weights_raw.json`, `objective_weights_stabilized.json`,
`table3_stabilized_weights.csv`, `stabilization_admissible_grid.csv`.

---

## Step 2 — Main experimental plans

```bash
python -m src.pipeline_V6_smart \
    --survey-dir data/survey_results \
    --graph      data/processed/strasbourg_multimodal.graphml \
    --output-dir results/outputs_main \
    --plans      main extended convergence ablation \
    --max-workers 3
```

Runs are checkpointed per `(profile, algorithm, seed)`; re-running the command
resumes rather than recomputing. Add `--profile-limit`, `--seed-limit` and
`--generations` for a fast smoke test.

---

## Step 3 — Analysis

```bash
python -m src.analytics_V6 --runs results/outputs_main --out results/analytics
```

---

## Tables

| Manuscript | Command | Output |
|---|---|---|
| **Table 2** — comfort surrogate validity | Step 2 | `results/outputs_main/comfort_model_comparison.csv` |
| **Table 3** — raw vs stabilized weights | Step 1 | `results/preferences/table3_stabilized_weights.csv` |
| **Table 4** — constraint handling | — | declarative; implemented in `src/optimization_framework_parallel3.py` (`PenaltyProblem` for MOEA/D, pymoo default elsewhere) |
| **Table 5** — strata distribution | Step 2 | `results/outputs_main/*_sampling_audit.csv` |
| **Table 6** — network descriptors | Step 0 | `results/network/table6_network_descriptors.csv` |
| **Table 7** — NSGA-II population sweep | `python -m experiments.popsize_equalization` | `results/experiments/popsize/table7_popsize_sweep.csv` |
| **Table 8** — nHV distribution + bootstrap CI | Step 3 | `results/analytics/main_*_table8_distribution.csv` |
| **Table 9** — nHV per archetype | Step 3 | `results/analytics/*_table9_per_archetype.csv` |
| **Table 10** — four-algorithm benchmark | Step 3 | `results/analytics/extended_*_table10_benchmark.csv` |
| **Table 11** — Friedman–Nemenyi pairwise | Step 3 | `results/analytics/extended_*_table11_nemenyi.csv` |
| **Table 12** — objective correlations | Step 3 | `results/analytics/table12_objective_correlations.csv` (+ `_spearman`) |
| **Table 13** — anchor-spread ρ sweep | `python -m experiments.rho_sweep` | `results/experiments/rho_sweep/table13_rho_sensitivity.csv` |
| **Table 14** — four-way ablation chain | `python -m experiments.four_way_ablation` | `results/experiments/four_way_ablation/table14_ablation_chain.csv` |
| **Table 15** — ablation pairwise | same command | `.../table15_pairwise.csv` |
| **Table 16** — wall-clock cost | `python -m experiments.anytime_analysis` | `results/experiments/anytime/table16_runtime_summary.csv` |
| **Table 17** — anytime hypervolume | same command with `--history-dirs results/outputs_main` | `.../table17_anytime_hypervolume.csv` |

## Figures

| Manuscript | Command | Output |
|---|---|---|
| **Fig. 1** — graph and path encoding | — | schematic, drawn in the manuscript source |
| **Fig. 2** — anchors on the reference lattice | `python -m src.reference_directions` (import `anchor_directions`) | reference sets for any ρ |
| **Fig. 3** — nHV distribution per plan | Step 3 | `results/analytics/fig_distribution_per_plan.png` |
| **Fig. 4** — per-profile caterpillar | Step 3 | `results/analytics/main_*_fig_caterpillar.png` |
| **Fig. 5** — per-profile *d_z* histogram | Step 3 | `results/analytics/main_*_fig_dz_distribution.png` |
| **Fig. 6** — Critical Difference diagram | Step 3 | `results/analytics/extended_*_fig_cd_diagram.png` |
| **Fig. 7** — HV convergence | Step 3 | `results/analytics/fig_hv_convergence.png` |
| **Fig. 9** — feasibility ratio per generation | Step 3 | `results/analytics/feasibility_ratio_per_generation.csv` |
| **Fig. 10** — intrinsic dimensionality per algorithm | `python -m experiments.intrinsic_dimensionality --runs results/outputs_main` | `results/analytics/intrinsic_dimensionality.json` |
| **Fig. 14** — surrogate comparison | Step 2 | `results/outputs_main/comfort_model_comparison.csv` |
| **Fig. 15** — *d_z* against ρ | `python -m experiments.rho_sweep` | `results/experiments/rho_sweep/table13_rho_sensitivity.csv` |
| **Fig. 16–17** — ablation distributions and *d_z* | `python -m experiments.four_way_ablation` | `results/experiments/four_way_ablation/` |
| **Fig. 18** — (β, φ) grid | `python -m experiments.beta_phi_grid` | `results/experiments/beta_phi_grid/fig18_dz_matrix.csv` |

## Sections without a numbered artefact

| Manuscript | Command |
|---|---|
| §5.3 population-size equalization | `python -m experiments.popsize_equalization` |
| §6.1 Holm correction over the 150 per-profile tests | Step 3 → `*_per_profile_paired_tests.csv` (`p_holm`, `significant_holm`) |
| §6.4 decoupling experiment | `python -m experiments.decoupling` |
| §6.6 surrogate ablation | `python -m experiments.surrogate_perturbations --study ablation` |
| §6.6 surrogate input noise | `python -m experiments.surrogate_perturbations --study noise` |
| §6.6 bias injection | `python -m experiments.surrogate_perturbations --study bias` |
| §6.6 Monte Carlo objectives | `python -m experiments.monte_carlo_objectives` |
| §6.6 normalization schemes | `python -m experiments.normalization_schemes --runs results/outputs_main/main_*` |

---

## Full sequence

```bash
python -m src.network.builder
python -m src.network.descriptors            --out results/network
python -m src.preferences.stabilization      --out results/preferences

python -m src.pipeline_V6_smart --output-dir results/outputs_main \
    --plans main extended convergence ablation --max-workers 3

python -m src.analytics_V6                   --runs results/outputs_main --out results/analytics
python -m experiments.intrinsic_dimensionality --runs results/outputs_main --out results/analytics

python -m experiments.popsize_equalization
python -m experiments.four_way_ablation
python -m experiments.rho_sweep
python -m experiments.beta_phi_grid
python -m experiments.decoupling
python -m experiments.surrogate_perturbations --study ablation
python -m experiments.surrogate_perturbations --study noise
python -m experiments.surrogate_perturbations --study bias
python -m experiments.monte_carlo_objectives
python -m experiments.normalization_schemes --runs results/outputs_main/main_pi_nsga3_vs_nsga2_150profiles
python -m experiments.anytime_analysis --history-dirs results/outputs_main
```

## Data flow

```
data/raw/  (OSM + GTFS)          data/survey_results/  (749 respondents)
        |                                 |
        v                                 v
src/network/builder.py            src/survey_data_loader.py
        |                                 |
        +--------> src/pipeline_V6_smart.py <--------+
                          |                          |
                          v                          v
              results/outputs_main/          src/preferences/stabilization.py
                          |
              +-----------+-----------+
              v                       v
      src/analytics_V6.py        experiments/*.py
              |                       |
              v                       v
      results/analytics/     results/experiments/
```

## Reported values versus manuscript values

`DEVIATIONS.md` lists every quantity where the value this code produces differs
from the value printed in the manuscript, with the reason. Read it before
citing any number from a fresh run.
