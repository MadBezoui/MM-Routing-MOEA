# DEVIATIONS.md

Every quantity where the value this code produces differs from the value
printed in the manuscript, with the reason. The manuscript was not modified;
this file records the gap so that it can be closed deliberately.

Three categories:

- **[DATA]** — the released empirical data contradicts the manuscript. Cannot
  be fixed in code without fabricating survey responses.
- **[RERUN]** — the code now implements what the manuscript describes, but the
  published numbers came from the superseded formulation. Re-running is
  required; the new values are not yet known at full scale.
- **[MODEL]** — a modelling choice in the construction that the manuscript does
  not pin down, where the value obtained depends on the choice.

---

## 1. Survey instrument (§4.1) — [DATA]

Run `python -m src.survey_data_loader` or read
`results/outputs_main/survey_instrument_report.json`. Measured on
`data/survey_results/04_comfort_scenario_ratings.csv`:

| Manuscript | Released data |
|---|---|
| "approximately **5,000** real trip-comfort pairs" | **8,988** |
| Table 2 caption "20 % held-out test set (**n = 1,000**)" | **1,800** (20 % of 8,988, respondent-level) |
| "each respondent evaluated between **4 and 9** scenarios (median 7)" | **exactly 12** for all 749 respondents |
| "**five-point** ordinal scale … normalized by (x−1)/4" | column `human_comfort_rating_1_10`, observed values 2–10, normalized as **x/10** |
| "respondents completing fewer than 4 scenarios were excluded" | no respondent falls below 12, so the filter never fires |

**Not fixable in code.** Changing any of these would mean altering recorded
responses. The manuscript text needs correcting, or a different survey export
needs releasing.

### Related but now fixed

- The manuscript says "no synthetic data generation is involved in comfort
  modelling". The previous loader sampled `rain` from `Binomial(1, 0.30)` and
  `temperature_c` from `N(14, 8)`, and set `age`, `mobility_restriction` and
  `safety_penalty` to constants. Now `rain` is derived from the recorded
  `weather` column, `age` and `mobility_restriction` are joined from the
  demographics file per respondent, and `temperature_c` is a deterministic map
  from `weather` to September climate normals. Nothing is sampled.
  `describe_survey()` reports `randomly_generated: []`.
- `generate_synthetic_dataset()` has been removed from `comfort_models.py`.

---

## 2. Network descriptors, Table 6 — [MODEL]

`results/network/table6_network_descriptors.csv`, recomputed from the graph:

| Descriptor | Manuscript | This build |
|---|---|---|
| Nodes \|V\| | 312 | **316** |
| Edges \|E\| | 1,847 | **5,033** |
| walk | 623 (33.7 %) | **1,496** (29.7 %) |
| bike | 412 (22.3 %) | **1,518** (30.2 %) |
| bus | 318 (17.2 %) | **573** (11.4 %) |
| tram | 189 (10.2 %) | **192** (3.8 %) |
| car | 305 (16.5 %) | **1,254** (24.9 %) |
| Mean node degree | 11.8 | **31.9** (2\|E\|/\|V\|); 5.79 on the simple graph |
| Mean shortest path (edges) | 6.2 | **8.64** |
| Feasible paths per OD pair | 47.3 mean / 31 median | **149** mean (enumeration capped at 8 edges, 400 paths) |
| Transfer hubs (≥ 3 modes) | 24 | **314** — see note |

Node count, tram and bus land close; walk, bike and car are 2.4–4× higher, so
\|E\| is 2.7× the published figure.

**Why the road layers are dense.** The manuscript states the preprocessing
qualitatively ("disconnected components removed, transfer hubs consolidated,
topological artifacts filtered") without giving parameters. Ours are recorded in
`data/processed/strasbourg_multimodal.build_config.json`. Two drive the counts:

- `min_trips_per_node = 24` — a facility is kept only if at least 24 trips serve
  it in the three-hour reference window, roughly one departure every
  7.5 minutes across all lines. This is the "topological artifacts filtered" step.
- `walk/bike/car_neighbours = 4/4/3` — each facility links to its *k* nearest
  reachable facilities in each road layer.

An earlier draft of this build used `k = (2, 2, 1)`, which gave \|E\| = 2,308
and edge counts within 1 % of Table 6 on walk, tram and car. **It was rejected
as structurally broken.** With k that low the k-nearest-neighbour construction
is close to acyclic — two facilities are mutually nearest only by coincidence —
and the largest strongly connected component of each road layer collapsed:

| Layer | k = (2, 2, 1) | k = (4, 4, 3) |
|---|---|---|
| walk | 251 / 313 nodes | **313 / 313** |
| bike | 282 / 309 | **309 / 309** |
| car | **11 / 312** | **312 / 312** |

At k = (2, 2, 1) a traveller could not walk back the way they came, no
car-dominated route family was realisable, and the walking-distance constraint
`W_lim,u` of Eq. 5 could never bind. `k = (4, 4, 3)` is the smallest setting at
which all three layers become strongly connected; denser settings shorten paths
only marginally. Matching Table 6 more closely would have meant shipping a
network that is not a transport network.

**Transfer hubs.** 314 of 316 nodes carry three or more modes, because the walk,
bike and car layers touch nearly every facility. The manuscript's 24 must use a
narrower definition. The descriptor file also reports
`n_interchange_hubs_bus_and_tram = 55`, the count of nodes where a traveller can
change between bus and tram without walking, which is the natural reading of
"transfer hub". Neither equals 24.

**What is definitely fixed.** The previous graph had 538,465 nodes,
1,570,533 edges and only three edge labels (`walk`, `transit`, `transfer`) —
no bike layer, no car layer, bus and tram merged. It could not support Table 6
under any definition. The new graph has all five modes, is strongly connected,
every road layer is strongly connected, and every descriptor is recomputed
rather than asserted.

---

## 3. Runtime, Table 16 and §5.4 — [RERUN]

The previously released `outputs_runtime_profiling/runtime_summary.csv` (now in
`results/legacy_continuous_pipeline/`) matched no cell of Table 16. It was also
produced by the superseded continuous formulation, so it could not.

`experiments/anytime_analysis.py` now measures the three costs the manuscript
distinguishes — instrumented, bare, and bare-truncated — on the real path-encoded
pipeline, single-process, and writes `table16_runtime_summary.csv` plus an
`environment.json` recording the machine.

**Table 16 must be re-measured on the target machine.** The MacBook Pro M3
figures in the manuscript cannot be reproduced on other hardware, and the search
being timed is now a different one. The derived claims in §7.3 — the 2.2×
instrumentation factor, the 1.15 s on-demand latency, the "under 12 minutes for
a 150-profile base" — all depend on this table.

---

## 4. Comfort surrogate, Table 2 — partially [RERUN]

Trained on the twelve features the manuscript lists, with min-max scaling,
median imputation on the training fold, a sigmoid output and early stopping on
a 20 % validation split:

| Model | Manuscript R² / RMSE / MAE | This code |
|---|---|---|
| Heuristic baseline | 0.42 / 0.115 / 0.094 | **0.473 / 0.108 / 0.081** |
| Linear regression | 0.64 / 0.090 / 0.073 | **0.658 / 0.087 / 0.070** |
| Multilayer perceptron | 0.69 / 0.084 / 0.068 | **0.692 / 0.083 / 0.062** |

The MLP reproduces the manuscript to two decimals. The heuristic baseline is
better than reported, which narrows the "survey calibration is necessary" gap
in §4.1 from 0.42→0.69 to 0.47→0.69.

### Input-feature noise, §6.6 — [RERUN]

The manuscript says noise is applied "to the twelve input features". The
previous implementation added noise to the **comfort label** and retrained,
which measures a different thing. `noise_robustness()` now perturbs the scaled
input features and leaves labels untouched. Consequence:

| σ | Manuscript RMSE / R² | This code |
|---|---|---|
| 0.00 | 0.084 / 0.69 | 0.0825 / 0.693 |
| 0.05 | 0.096 / 0.61 | **0.0822 / 0.696** |
| 0.10 | 0.115 / 0.50 | **0.0828 / 0.691** |

The surrogate is far more robust to input noise than the manuscript reports.
Degradation only becomes visible above σ ≈ 0.3 (R² 0.59) and σ ≈ 0.6 (R² 0.24).
The §6.6 paragraph needs rewriting against the measured curve.

---

## 5. Survey-derived statistics, §5.2 — [DATA]

| Manuscript | Computed from `01_demographics_profiles.csv` |
|---|---|
| OD distance / budget correlation r = 0.70 (p < 0.01) | **r = 0.846** |
| OD distance std 3.2 km | **3.04 km** (mean 4.86 ✓) |

---

## 6. Emission factor, §3.2 — fixed

The manuscript quotes tram ≈ 40 g CO₂/km; the previous code used 35. Now 0.040
kg/km in `src/network/evaluator.py`, matching the manuscript. Bus 80, car 150,
walk and bike 0 were already correct.

---

## 7. Inter-objective correlation structure, Table 12 and §3.4(iv) — [RERUN]

Measured on the feasible archive of a path-encoded run:

| Pair | Manuscript | Path problem |
|---|---|---|
| Time – Cost | +0.69 | **+0.43** |
| Time – Emissions | +0.55 | **+0.33** |
| Time – Discomfort | −0.53 | **+0.13** — sign flip |
| Cost – Emissions | +0.47 | **+0.93** |
| Cost – Discomfort | −0.48 | **−0.08** |
| Emissions – Discomfort | −0.24 | **−0.27** |

Two claims that rest on this table no longer hold as written:

- §3.4(iv) states the correlations are moderate, `|r| ∈ [0.24, 0.69]`. The
  observed range is `[0.08, 0.93]`.
- §6.4 states that "time, cost and emissions form a positively correlated
  triple" and that discomfort is negatively correlated with all three. Time and
  discomfort are now weakly *positively* correlated.

The change is mechanical: on a real network the car layer is fast, expensive and
high-emission while transit is slower, cheaper and cleaner, so cost and
emissions become nearly collinear (+0.93) and the time-versus-comfort trade-off
loses its sign. This is a property of the path problem, not a bug.

The intrinsic-dimensionality story survives: TwoNN on the pooled archive gives
**1.64** against the manuscript's 1.56. The Levina-Bickel cross-check gives
**1.11** against the reported 2.63, so the "TwoNN below Levina-Bickel" ordering
in Fig. 10 is reversed. All of this is at smoke scale and must be recomputed on
the full plan.

---

## 8. All headline results — [RERUN]

Every number in Tables 8–15 and Figures 3–9 was produced by the superseded
continuous mode-share formulation. Those files are preserved under
`results/legacy_continuous_pipeline/` and are **fully reproducible from
themselves** — the audit re-derived Table 8, the paired *d_z* of −1.03 and
−1.27, W = 48, p = 6.0 × 10⁻²⁶, 145/150 profiles, Tables 9–12 and the
Friedman–Nemenyi report exactly from those CSVs.

But they do not describe the problem of Section 3. Re-running Step 2 of
`REPRODUCE.md` on the path-encoded problem will produce different values.
Nothing in this repository currently claims otherwise.

### The headline direction is not yet confirmed

On the smoke-scale runs used to validate the code — 3 profiles, 2 seeds,
12 generations, which is far too small to conclude anything — NSGA-II came out
**ahead** of PI-NSGA-III on the main plan (mean nHV 0.9835 against 0.9783) and
on the convergence plan (0.9943 against 0.9713), and behind it on the extended
benchmark (0.9928 against 0.9943). The four-way ablation chain did not order
itself monotonically either.

These numbers mean nothing statistically. They are recorded here for one
reason: **do not assume the −0.0189 mean gap and *d_z* = −1.27 of Section 6.1
will reappear.** The problem being solved has changed. The full 150-profile,
30-seed, 150-generation run has to be executed before Section 6.1 can be
restated, and its outcome is genuinely unknown.

What the search itself does look healthy:

- feasibility ratio climbs from 54.6 % at generation 1 to 100 % by generation 10,
  so constrained domination is doing work rather than sitting idle;
- all three terms of Eq. 5 bind on initial populations — budget 4.7 %,
  travel time 33.9 %, walking distance 20.0 % of sampled routes violate them;
- routes carry a median of 1 transfer and 16 edges, and all five mode-dominated
  families appear in the initial population;
- objective ranges are plausible: travel time 26–79 min, cost €0.3–6.8,
  emissions 0–2.7 kg CO₂ (Section 3.2 predicts `f₃ ∈ [0, 3]`), discomfort
  0–0.17.

---

## 9. Things the manuscript describes that now exist and did not before

Not deviations — closed gaps, listed so the diff is auditable.

| Manuscript element | Status before | Now |
|---|---|---|
| Path/graph problem (§3.1, Eq. 1–3) | continuous 5-variable mode-share problem | `src/network/evaluator.py`, edge-summed over a real graph |
| Transfer penalty 3–15 min | fixed 4 min | quality-interpolated, headway-derived for transit boardings |
| Eq. 5, three normalized terms | two unnormalized terms, walking bound global | three terms, each normalized, all three bounds per profile |
| `T_max,u` travel-time constraint | absent | `max_travel_time_min` per respondent |
| `W_lim,u` walking bound | sample mean for all profiles | `max_walking_distance_m` per respondent |
| Path crossover / mutation (§4.3) | SBX + polynomial mutation | `src/network/operators.py`, suffix swap + parallel-edge/detour |
| Topological validity vs operational feasibility | not implemented | `src/network/route.py` |
| Anchor spread ρ | hardcoded `0.7*w + 0.3*e` | parameter, swept by `experiments/rho_sweep.py` |
| Canonical NSGA-III baseline | received the priority-informed set | plain Das-Dennis, `src/reference_directions.py` |
| Four-way ablation (Tables 14–15) | no code | `experiments/four_way_ablation.py` |
| (β, φ) grid (Fig. 18) | no code | `experiments/beta_phi_grid.py` |
| Eq. 7 admissible set | no code | `admissible_pairs()` in `stabilization.py` |
| Decoupling experiment (§6.4) | no code | `experiments/decoupling.py` |
| Bias injection (§6.6) | no code | `experiments/surrogate_perturbations.py --study bias` |
| Monte Carlo objectives (§6.6) | no code | `experiments/monte_carlo_objectives.py` |
| Normalization schemes (§6.6) | no code | `experiments/normalization_schemes.py` |
| Levina-Bickel MLE (Fig. 10) | no code | `experiments/intrinsic_dimensionality.py` |
| Anytime table (Table 17) | no code | `experiments/anytime_analysis.py` |
| Holm correction (§6.1) | no code | `src/statistics.py`, in every per-profile table |
| Profile-stratified bootstrap CI (Table 8) | no code | `stratified_bootstrap_ci()` |
| Seeds: 30 main / 10 extended | 20 everywhere | `config.SEEDS`, enforced |
| `plan_type` wiring | never passed; all plans ran as "main" | passed through `execute_plan` |
| Per-mode adjacency, degree histogram (§5.2) | not released | `results/network/` |
| OD assignment | random node pairs | matched to each respondent's reported commuting distance |

### Defects found and fixed during verification

These were introduced by this rewrite and caught by the double-check pass; they
are listed so the review trail is complete.

| Defect | Symptom | Fix |
|---|---|---|
| `PathCrossover` applied its probability twice | pymoo masks matings with the base-class `prob`, and `_do` tested it again — effective p_c was 0.81, not the 0.9 stated in §4.3 | probability delegated to the base class only |
| Operators drew from the global `numpy.random` | pymoo 0.6.2 threads a per-run `Generator` through `random_state`; ignoring it made concurrent runs in the thread pool non-reproducible | all three operators use the injected generator; two runs at three workers are now bit-identical |
| Sampler ignored the transfer penalty | routes averaged 8.2 mode changes, waiting was 73 % of travel time, median trip 83 min | Dijkstra runs over `(node, mode)` states and charges the interchange |
| Sampler had no mode preference | walking and car never appeared; only bus- and bike-dominated routes | per-individual mode family with a discount/surcharge, cycling through the five families of §3.4(i) |
| Road layers were near-acyclic | see §2 above | road links made bidirectional, `k` raised to (4, 4, 3) |
