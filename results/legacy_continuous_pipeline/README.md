# Legacy results — superseded continuous formulation

These files were produced by an earlier version of the pipeline that solved a
**continuous five-variable mode-share allocation problem**, not the path-based
problem of Section 3 of the manuscript. In that version:

- decision variables were five real numbers normalised into mode shares, with
  no graph, no node sequence and no edge traversal;
- travel time was a share-weighted speed plus a flat four-minute penalty per
  transfer, rather than a sum over edges with quality-dependent waits;
- variation used SBX and polynomial mutation, not the path operators of §4.3;
- constraint violation had two unnormalized terms (budget, walking) instead of
  the three normalized terms of Eq. 5, and the walking bound was the sample
  mean rather than each respondent's own limit.

**Every number in Tables 8 to 15 and Figures 3 to 9 of the manuscript comes
from these files.** They are internally consistent: the headline statistics —
mean nHV 0.9930 against 0.9741, run-level *d_z* = −1.03, profile-level
*d_z* = −1.27, W = 48, p = 6.0 × 10⁻²⁶, 145 of 150 profiles, Tables 9 to 12 and
the Friedman–Nemenyi report — all re-derive exactly from the CSV files in this
directory.

They are kept for provenance. They are **not** the output of the code now in
`src/`, and re-running the pipeline will produce different values.

## Contents

| Path | What it is |
|---|---|
| `main_nsga2_vs_nsga3_150profiles_final_generation_recovered.csv` | 9,000 runs, 150 profiles × 30 seeds × 2 algorithms |
| `extended_benchmark_30profiles_final_generation_recovered.csv` | 1,050 runs, four algorithms |
| `representative_curves_10profiles_final_generation_recovered.csv` | convergence plan |
| `analytics/` | derived tables and figures |
| `comfort_*.csv` | comfort surrogate metrics under the previous feature set |
| `objective_weights_*.json` | raw and stabilized priority weights — these **do** match the current code exactly |
| `outputs_popsize_equalization/` | Table 7 population sweep |
| `outputs_runtime_profiling/` | runtime measurements; these match no cell of Table 16 |
| `outputs_surrogate/` | contains **dummy** placeholder weights `{0.1, 0.1, 0.05, 0.75}`, not survey-derived |

See `../../DEVIATIONS.md` for the full comparison.
