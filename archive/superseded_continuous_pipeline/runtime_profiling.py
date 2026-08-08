"""runtime_profiling.py
====================
Per-profile latency profiling.

The per-run times produced by the benchmark are measured with the full
experimental instrumentation active: the MetricsCallback recomputes
hypervolume, IGD and spacing at EVERY generation.  That instrumentation is a
measurement artefact of the benchmark, not a cost a deployed MaaS platform
would pay.  This script separates the two, so that the deployment cost can be
stated in measured numbers.

It measures, on the main-plan operating point (N=170, 150 generations):

  A. instrumented   -- per-generation HV/IGD/spacing callback active
                       (reproduces the benchmark measurement)
  B. bare           -- optimization only, no per-generation indicators
                       (the deployment cost)
  C. truncated      -- bare, with the generation budget reduced to the point
                       where HV has plateaued (see --trunc-gens)

Usage
-----
    python runtime_profiling.py --n-profiles 10 --n-reps 5
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent / "src"))

from config import DEFAULT_COMFORT_CONFIG, DEFAULT_SCENARIO
from comfort_models import SurveyInformedComfortFactory
from pipeline_V6_smart import (
    audit_and_stabilize_weights,
    balanced_sample_by_group,
    build_reference_point_factory,
    TrainedComfortPredictor,
    build_problem_factory,
)
from optimization_framework_parallel3 import (
    MetricsCallback,
    weighted_reference_directions,
)
from survey_data_loader import SurveyData, load_all

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

N_OBJ = 4


def build(algo_name: str, pop: int, ref_dirs):
    common = dict(sampling=FloatRandomSampling(),
                  crossover=SBX(prob=0.9, eta=15.0),
                  mutation=PM(eta=20.0))
    if algo_name == "nsga2":
        return NSGA2(pop_size=pop, **common)
    return NSGA3(pop_size=pop, ref_dirs=ref_dirs, **common)


def timed(problem, algo_name, pop, ref_dirs, seed, n_gen, instrumented, ref_point):
    algo = build(algo_name, pop, ref_dirs)
    cb = MetricsCallback(reference_front=None, reference_point=ref_point) if instrumented else None
    t0 = time.perf_counter()
    kwargs = {"seed": seed, "verbose": False, "save_history": False}
    if cb is not None:
        kwargs["callback"] = cb
    minimize(problem, algo, ("n_gen", n_gen), **kwargs)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey-dir", default="data/survey_results")
    ap.add_argument("--out-dir", default="outputs_runtime_profiling")
    ap.add_argument("--n-profiles", type=int, default=10)
    ap.add_argument("--n-reps", type=int, default=5)
    ap.add_argument("--n-gen", type=int, default=150)
    ap.add_argument("--trunc-gens", type=int, default=40)
    ap.add_argument("--partitions", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    survey: SurveyData = load_all(Path(args.survey_dir))
    real_survey = survey.calibration
    audit = audit_and_stabilize_weights(survey.objective_weights)
    pw = [audit.stabilized_weights.get(k, 0.25)
          for k in ("time", "cost", "emissions", "comfort")]

    factory = SurveyInformedComfortFactory(DEFAULT_COMFORT_CONFIG, real_survey)
    tdf = survey.comfort_training.copy()
    factory.generate_synthetic_dataset = lambda n_samples=None, seed=None: tdf.copy()
    predictor = TrainedComfortPredictor(factory.train_models(tdf))
    problem_factory = build_problem_factory(real_survey, predictor)
    ref_point_factory = build_reference_point_factory(real_survey)

    main_profiles = balanced_sample_by_group(
        survey.profiles.copy(), sample_size=150,
        group_cols=["archetype", "trip_distance_bin"],
        random_state=42, force_min_per_group=1)
    subset = balanced_sample_by_group(
        main_profiles, sample_size=args.n_profiles,
        group_cols=["archetype", "trip_distance_bin"],
        random_state=57, force_min_per_group=1)

    ref_dirs = weighted_reference_directions(N_OBJ, args.partitions, priority_weights=pw)

    conditions = [
        ("nsga3", 170, args.n_gen, True,  "instrumented"),
        ("nsga3", 170, args.n_gen, False, "bare"),
        ("nsga3", 170, args.trunc_gens, False, "bare_truncated"),
        ("nsga2", 168, args.n_gen, True,  "instrumented"),
        ("nsga2", 168, args.n_gen, False, "bare"),
    ]

    rows: List[Dict] = []
    for _, prof in subset.iterrows():
        rec = prof.to_dict()
        problem = problem_factory(rec, DEFAULT_SCENARIO)
        rp = ref_point_factory(rec)
        for algo, pop, ngen, instr, tag in conditions:
            for rep in range(args.n_reps):
                dt = timed(problem, algo, pop, ref_dirs, rep, ngen, instr, rp)
                rows.append(dict(profile_id=rec.get("profile_id"),
                                 archetype=rec.get("archetype"),
                                 algorithm=algo, pop_size=pop, n_gen=ngen,
                                 instrumented=instr, condition=tag,
                                 rep=rep, seconds=dt))
        pd.DataFrame(rows).to_csv(out / "runtime_raw.csv", index=False)
        print(f"[rt] {rec.get('profile_id')} done ({len(rows)} timings)", flush=True)

    df = pd.DataFrame(rows)
    summary = (df.groupby(["algorithm", "condition", "n_gen"])["seconds"]
                 .agg(["count", "mean", "median", "std",
                       lambda s: np.percentile(s, 95)])
                 .rename(columns={"<lambda_0>": "p95"})
                 .reset_index())
    summary.to_csv(out / "runtime_summary.csv", index=False)

    json.dump({"machine": platform.platform(),
               "processor": platform.processor(),
               "python": platform.python_version(),
               "note": "single thread, no concurrent worker threads"},
              open(out / "environment.json", "w"), indent=2)

    print(summary.to_string(index=False))
    print(f"[rt] DONE -> {out}")


if __name__ == "__main__":
    main()
