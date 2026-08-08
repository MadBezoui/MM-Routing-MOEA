"""popsize_equalization.py
=========================
Population-size equalization and sweep experiments.

NSGA-II runs at the configured population size N=168, while PI-NSGA-III is
raised to N=170 to match the cardinality of the augmented reference set.  Two
questions follow, and both are answered here with real runs:

(Q1) EQUALIZATION. Does the 168 vs 170 asymmetry of the main plan explain the
     PI-NSGA-III advantage?  -> Re-run BOTH algorithms at a COMMON population
     size on a stratified profile subset, and also reproduce the asymmetric
     configuration of the main plan on the same subset as a control.

(Q2) SWEEP. Is N=170 in any way special for NSGA-II?  -> Run NSGA-II across a
     grid of population sizes on the same subset and inspect the nHV profile.

The hypervolume protocol reproduces the one used in the paper exactly:
  * per profile, the nadir is the component-wise maximum over the UNION of all
    feasible final-population members of all runs of that profile, plus 1e-6;
  * a run's HV uses all feasible members of its final population;
  * nHV = HV / max(HV over all runs of that profile)
    ("union_observed_max_per_profile").

Usage
-----
    python popsize_equalization.py --n-profiles 30 --n-seeds 15
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from config import DEFAULT_COMFORT_CONFIG, DEFAULT_SCENARIO
from comfort_models import SurveyInformedComfortFactory
from pipeline_V6_smart import (
    audit_and_stabilize_weights,
    balanced_sample_by_group,
    TrainedComfortPredictor,
    build_problem_factory,
)
from optimization_framework_parallel3 import weighted_reference_directions
from survey_data_loader import SurveyData, load_all

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV

N_OBJ = 4


def make_algo(name: str, pop_size: int, ref_dirs,
              cx_prob: float = 0.9, cx_eta: float = 15.0, mut_eta: float = 20.0):
    common = dict(
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=cx_prob, eta=cx_eta),
        mutation=PM(eta=mut_eta),
    )
    if name == "nsga2":
        return NSGA2(pop_size=pop_size, **common)
    if name == "nsga3":
        # pymoo raises pop_size to len(ref_dirs) when it is smaller; passing it
        # explicitly keeps the effective population size unambiguous.
        return NSGA3(pop_size=pop_size, ref_dirs=ref_dirs, **common)
    raise ValueError(name)


def run_one(problem, algo_name: str, pop_size: int, ref_dirs, seed: int, n_gen: int):
    """Run one (profile, algorithm, N, seed) and return the feasible final
    population in objective space, plus bookkeeping."""
    algo = make_algo(algo_name, pop_size, ref_dirs)
    t0 = time.perf_counter()
    res = minimize(problem, algo, ("n_gen", n_gen), seed=seed,
                   verbose=False, save_history=False)
    elapsed = time.perf_counter() - t0

    F = np.asarray(res.pop.get("F"), dtype=float)
    G = res.pop.get("G")
    if G is None:
        feasible = np.ones(len(F), dtype=bool)
    else:
        G = np.asarray(G, dtype=float)
        feasible = np.all(G <= 0, axis=1) if G.ndim > 1 else (G <= 0)

    return dict(
        F_feasible=F[feasible],
        effective_pop=int(len(F)),
        n_feasible=int(feasible.sum()),
        runtime_s=float(elapsed),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey-dir", default="data/survey_results")
    ap.add_argument("--out-dir", default="outputs_popsize_equalization")
    ap.add_argument("--n-profiles", type=int, default=30)
    ap.add_argument("--n-seeds", type=int, default=15)
    ap.add_argument("--n-gen", type=int, default=150)
    ap.add_argument("--partitions", type=int, default=8)
    ap.add_argument("--equal-n", type=int, default=170)
    ap.add_argument("--sweep", default="120,140,150,160,168,170,180,200")
    ap.add_argument("--sweep-profiles", type=int, default=10)
    ap.add_argument("--sweep-seeds", type=int, default=10)
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--time-budget", type=float, default=0.0,
                    help="stop gracefully after this many seconds (0 = no limit); "
                         "already-completed profiles are reloaded on the next call")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[eq] loading survey data ...", flush=True)
    survey: SurveyData = load_all(Path(args.survey_dir))
    real_survey = survey.calibration
    profiles_df = survey.profiles.copy()
    training_df = survey.comfort_training.copy()

    audit = audit_and_stabilize_weights(survey.objective_weights)
    priority_weights = [audit.stabilized_weights.get(k, 0.25)
                        for k in ("time", "cost", "emissions", "comfort")]

    print("[eq] training comfort surrogate ...", flush=True)
    factory = SurveyInformedComfortFactory(DEFAULT_COMFORT_CONFIG, real_survey)
    factory.generate_synthetic_dataset = lambda n_samples=None, seed=None: training_df.copy()
    comfort_predictor = TrainedComfortPredictor(factory.train_models(training_df))
    problem_factory = build_problem_factory(real_survey, comfort_predictor)

    # Same stratified recipe as the paper's extended benchmark subset.
    main_profiles = balanced_sample_by_group(
        profiles_df, sample_size=150, group_cols=["archetype", "trip_distance_bin"],
        random_state=42, force_min_per_group=1)
    subset = balanced_sample_by_group(
        main_profiles, sample_size=args.n_profiles,
        group_cols=["archetype", "trip_distance_bin"],
        random_state=57, force_min_per_group=1)
    subset.to_csv(out / "equalization_profiles.csv", index=False)

    ref_dirs = weighted_reference_directions(N_OBJ, args.partitions,
                                             priority_weights=priority_weights)
    n_das_dennis = len(weighted_reference_directions(N_OBJ, args.partitions, None))
    print(f"[eq] |Das-Dennis| = {n_das_dennis}, |augmented| = {len(ref_dirs)}", flush=True)
    json.dump({"n_ref_dirs": int(len(ref_dirs)),
               "n_das_dennis": int(n_das_dennis),
               "n_anchors": int(len(ref_dirs) - n_das_dennis),
               "priority_weights": list(map(float, priority_weights)),
               "n_generations": args.n_gen,
               "n_profiles": int(len(subset)),
               "n_seeds": args.n_seeds},
              open(out / "reference_direction_audit.json", "w"), indent=2)

    # ------------------------------------------------------------------ plan --
    jobs: List[Dict] = []
    for algo, pop, tag in [("nsga2", args.equal_n, "equalized"),
                           ("nsga3", args.equal_n, "equalized"),
                           ("nsga2", 168, "as_published"),
                           ("nsga3", 170, "as_published")]:
        for seed in range(args.n_seeds):
            jobs.append(dict(experiment="equalization", algorithm=algo,
                             pop_size=pop, config=tag, seed=seed))
    if not args.skip_sweep:
        sweep_ids = set(subset["profile_id"].head(args.sweep_profiles))
    else:
        sweep_ids = set()
    sweep_sizes = [int(s) for s in args.sweep.split(",")]

    # --- resume support: reload profiles already completed in a previous call --
    csv_path = out / "equalization_raw.csv"
    rows: List[Dict] = []
    done_profiles = set()
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        rows = prev.to_dict(orient="records")
        done_profiles = set(prev["profile_id"].unique())
        print(f"[eq] resuming: {len(done_profiles)} profiles already done", flush=True)

    t_start = time.perf_counter()
    n_profiles = len(subset)

    for p_idx, (_, prof) in enumerate(subset.iterrows(), start=1):
        record = prof.to_dict()
        pid = record.get("profile_id")
        if pid in done_profiles:
            continue
        if args.time_budget and (time.perf_counter() - t_start) > args.time_budget:
            print(f"[eq] time budget reached; {n_profiles - len(done_profiles)} "
                  f"profiles remaining. Re-run to continue.", flush=True)
            return
        problem = problem_factory(record, DEFAULT_SCENARIO)

        profile_jobs = list(jobs)
        if pid in sweep_ids:
            for pop in sweep_sizes:
                for seed in range(args.sweep_seeds):
                    profile_jobs.append(dict(experiment="nsga2_sweep",
                                             algorithm="nsga2", pop_size=pop,
                                             config="sweep", seed=seed))

        results, fronts = [], []
        for job in profile_jobs:
            r = run_one(problem, job["algorithm"], job["pop_size"], ref_dirs,
                        job["seed"], args.n_gen)
            fronts.append(r.pop("F_feasible"))
            results.append({**job, **r})

        # --- paper-identical HV protocol, per profile -------------------------
        union = np.vstack([f for f in fronts if len(f)]) if any(len(f) for f in fronts) else None
        if union is None:
            print(f"[eq] profile {pid}: no feasible solution, skipped", flush=True)
            continue
        nadir = union.max(axis=0) + 1e-6
        hv_ind = HV(ref_point=nadir)

        hvs = [float(hv_ind(f)) if len(f) else np.nan for f in fronts]
        max_hv = np.nanmax(hvs) if np.any(~np.isnan(hvs)) else np.nan
        for res, hv in zip(results, hvs):
            res["profile_id"] = pid
            res["archetype"] = record.get("archetype")
            res["hypervolume"] = hv
            res["max_hv_profile"] = max_hv
            res["normalized_hv"] = hv / max_hv if max_hv and max_hv > 0 else np.nan
            res["normalization_scheme"] = "union_observed_max_per_profile"
            rows.append(res)

        pd.DataFrame(rows).to_csv(csv_path, index=False)
        done_profiles.add(pid)
        el = time.perf_counter() - t_start
        print(f"[eq] profile {p_idx}/{n_profiles} ({pid}) done "
              f"| total complete={len(done_profiles)}/{n_profiles} "
              f"| elapsed this call={el/60:.1f}min", flush=True)

    print(f"[eq] DONE -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
