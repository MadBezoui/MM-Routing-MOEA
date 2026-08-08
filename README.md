# Priority-informed NSGA-III for personalized multimodal route planning

Reference implementation for *Priority-informed many-objective optimization for
personalized multimodal route planning under survey-calibrated user
preferences* (Ibnelbey & Bezoui).

The framework formulates personalized multimodal routing as a **constrained
four-objective problem** — travel time, monetary cost, carbon emissions and a
survey-calibrated discomfort score — over a real OpenStreetMap/GTFS network of
the Strasbourg metropolitan area, and solves it with a priority-informed
variant of NSGA-III whose reference directions are anchored on stabilized user
priority weights.

---

## Layout

```
data/
  raw/                    frozen OSM extract, GTFS feed, study boundary, checksums
  survey_results/         749 respondents, five semicolon-separated files
  processed/              consolidated multimodal graph (built by src/network/builder.py)

src/
  config.py               every hyperparameter quoted in the manuscript, once
  survey_data_loader.py   survey -> profiles, comfort training set, elicited weights
  comfort_models.py       heuristic / linear / MLP comfort surrogates (Section 4.1)
  preferences/
    stabilization.py      Eq. 6-7: weight stabilization and the admissible set
  reference_directions.py Eq. 8-9: Das-Dennis lattice and priority anchors
  network/
    osm_parser.py         streaming OSM reader, three mode-specific road layers
    builder.py            consolidated five-mode multimodal graph (Section 5.2)
    descriptors.py        Table 6 descriptors, adjacency matrices, degree histogram
    route.py              solution encoding, topological validity (Section 3.1, 4.3)
    evaluator.py          Eq. 1-5: objectives and constraint violation
    operators.py          path sampling, crossover, mutation (Section 4.3)
  optimization_framework_parallel3.py   algorithm construction, parallel runner
  pipeline_V6_smart.py    orchestrator for the four experimental plans
  statistics.py           Section 4.5 protocol: paired tests, Holm, bootstrap, Friedman
  analytics_V6.py         Tables 8-12 and the results figures

experiments/              the sensitivity and ablation studies of Sections 6.4-6.6
results/
  network/                Table 6 artefacts
```

---

## What the code implements

**Problem (Section 3).** A solution is a path `P = (v₁,…,v_k)` on a directed
multimodal graph together with a mode sequence `M(P)`. Objectives are summed
over edges: travel time with transfer penalties of three to fifteen minutes
depending on the quality of the interchange (Eq. 1), per-mode tariffs with a
fare-variability factor (Eq. 2), length times per-mode emission factor times an
occupancy multiplier (Eq. 3), and one minus the surrogate comfort score
(Eq. 4). Feasibility aggregates three normalized violations — budget, maximum
travel time and walking distance — each bound taken **per profile** from that
respondent's own answers (Eq. 5).

**Method (Section 4).** The comfort surrogate is a two-hidden-layer perceptron
with a sigmoid output, trained on real trip-comfort ratings with a
respondent-level split so that `R²` measures generalisation to unseen people.
Elicited priority weights are stabilized by blending towards uniform and
applying a per-component floor, which prevents a near-zero elicited weight
(here emissions, at 8.9 × 10⁻⁵) from silently removing an objective. The
stabilized vector then anchors `M + 1` extra reference directions on the
Das-Dennis lattice.

**Evaluation (Section 4.4–4.5).** Hypervolume is normalized against a reference
point and a denominator built from the union of *all* algorithms and seeds of a
profile, so the measurement instrument favours no algorithm. The unit of
confirmatory inference is the profile, never the seed.

---

## Quick start

```bash
pip install numpy pandas scipy matplotlib scikit-learn pymoo networkx tqdm

# 1. build the network (once; parses the OSM extract and caches it)
python -m src.network.builder
python -m src.network.descriptors --out results/network

# 2. smoke test: 3 profiles, 2 seeds, 10 generations
python -m src.pipeline_V6_smart --output-dir /tmp/smoke \
    --plans main --profile-limit 3 --seed-limit 2 --generations 10

# 3. full run
python -m src.pipeline_V6_smart --output-dir results/outputs_main \
    --plans main extended convergence ablation --max-workers 3
python -m src.analytics_V6 --runs results/outputs_main --out results/analytics
```

`REPRODUCE.md` maps every manuscript table and figure to its command.

Runs are checkpointed per `(profile, algorithm, seed)`, so an interrupted
experiment resumes where it stopped.

---

## Experimental plans (Section 5.3)

| Plan | Profiles | Algorithms | Seeds | Generations | Population |
|---|---|---|---|---|---|
| main | 150 | NSGA-II, PI-NSGA-III | 30 | 150 | 168 / 170 |
| extended | 30 | + MOEA/D, SMS-EMOA | 10 (5 for SMS-EMOA) | 120 | 128 |
| convergence | 10 | NSGA-II, PI-NSGA-III | 30 | 150 | 168 / 170 |
| ablation | 30 | NSGA-II, canonical NSGA-III, PI-raw, PI-stab | 10 | 150 | 168 / 165 / 170 / 170 |

Population sizes are declared explicitly in `config.POPULATION_SIZES`. For the
reference-direction methods the value equals the cardinality of the reference
set, so that every direction is associated with at least one individual; it is
fixed by the construction, not tuned. `experiments/popsize_equalization.py`
quantifies what the 168-versus-170 asymmetry is worth.

---

## Frozen data inputs

- OpenStreetMap / Geofabrik Alsace snapshot: 2026-01-01
- Study boundary: Eurométropole de Strasbourg, EPCI 246700488
- Archived CTS GTFS feed: 2026-08-05
- Reference service date: 2026-09-15, departure 08:00 Europe/Paris

```bash
docker build -f Dockerfile.data -t pi-nsga3-data .
docker run --rm -v "${PWD}:/workspace" pi-nsga3-data
```

Source URLs and SHA-256 checksums are recorded in `data/raw/sources.json` and
`data/raw/checksums.sha256`.

> [!NOTE]
> The uncompressed `data/raw/strasbourg.osm` file (346 MB) exceeds GitHub's 100 MB file size limit and is therefore not tracked in the repository. Please download the original snapshot ([alsace-260101.osm.pbf](https://download.geofabrik.de/europe/france/alsace-260101.osm.pbf)) and extract it.

---

## Reproducibility and Results

This repository contains the exact reference implementation used to generate the results in the manuscript. All algorithmic variants, multimodal data processing scripts, and statistical evaluation tools (including the stabilized PI-NSGA-III formulation) are provided.

The codebase solves the discrete path-based routing formulation as formally described in Section 3 of the final manuscript.

## Verification status

| Check | Result |
|---|---|
| Formulation alignment | fully aligned with the manuscript's Section 3 (discrete path-based formulation) |
| Reproducibility | strict Common Random Numbers (CRN) stability enforced across processes |
| Constraint activity | feasibility robustly handled; all three Eq. 5 terms bind |
| Reference sets | canonical 165 directions, priority-informed 170 = 165 + 5, extended 120/125 |
| Priority weights | reproduce Table 3 to all printed digits |
| Comfort surrogate | R² = 0.693 against the reported 0.69 |
| Route realism | median 1 transfer, 16 edges; all five mode families present |
