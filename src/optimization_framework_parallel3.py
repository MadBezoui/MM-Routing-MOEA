"""optimization_framework_parallel3.py
======================================
Algorithm construction and parallel execution of the experimental plans.

The five algorithms of Table 4 are built here.  Constraint handling follows the
table exactly: feasibility-first constrained domination (pymoo's default) for
NSGA-II, PI-NSGA-III, canonical NSGA-III and SMS-EMOA, and a penalised
Tchebycheff scalarization with coefficient :math:`10^3` for MOEA/D.

Reference directions are delegated to :mod:`src.reference_directions`, which is
what keeps *canonical* NSGA-III genuinely canonical: it receives the plain
Das-Dennis lattice, never the priority-informed one.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.config import Config
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from tqdm.auto import tqdm

from src.config import (
    DEFAULT_ALGO_SWEEP,
    DEFAULT_EXPERIMENT,
    DEFAULT_REFDIRS,
    canonical_algorithm,
    resolve_population_size,
)
from src.reference_directions import build_reference_directions

Config.warnings["not_compiled"] = False
logger = logging.getLogger(__name__)

ObjectiveEvaluator = Callable[
    [np.ndarray, Dict[str, object], Dict[str, object], object],
    Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]],
]

#: Algorithms driven by a reference-direction / weight-vector set.
REF_DIR_ALGORITHMS = frozenset({
    "pi_nsga3", "pi_nsga3_raw", "pi_nsga3_stab", "canonical_nsga3", "moead",
})


@dataclass
class AlgorithmRunOutput:
    profile_id: str
    algorithm: str
    seed: int
    runtime_sec: float
    final_population: pd.DataFrame
    history: pd.DataFrame


# --------------------------------------------------------------------------
# Problems
# --------------------------------------------------------------------------

class ProfiledMultimodalProblem(Problem):
    """Single-profile instance of the constrained four-objective problem."""

    def __init__(
        self,
        n_var: int,
        n_obj: int,
        xl: Optional[Sequence[float]],
        xu: Optional[Sequence[float]],
        evaluator: ObjectiveEvaluator,
        profile: Dict[str, object],
        extras: Dict[str, object],
        scenario: object,
        n_ieq_constr: int = 1,
        vtype: type = float,
    ):
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_ieq_constr=n_ieq_constr,
            xl=None if xl is None else np.asarray(xl),
            xu=None if xu is None else np.asarray(xu),
            vtype=vtype,
        )
        self._evaluator = evaluator
        self.profile = profile
        self.extras = extras
        self.scenario = scenario
        #: Multimodal graph and its cached adjacency index, set by the problem
        #: factory when the path encoding is used.
        self.graph = None
        self.graph_index = None
        self.latest_meta: Dict[str, np.ndarray] = {}

    def _evaluate(self, X, out, *args, **kwargs):
        F, G, meta = self._evaluator(X, self.profile, self.extras, self.scenario)
        out["F"] = np.asarray(F, dtype=float)
        out["G"] = np.asarray(G, dtype=float)
        self.latest_meta = meta


class PenaltyProblem(Problem):
    """Constraint handling for MOEA/D: penalised Tchebycheff scalarization.

    The aggregated violation is multiplied by ``penalty_scale`` (:math:`10^3`,
    Table 4) and added to every objective, since MOEA/D as implemented in pymoo
    optimises an unconstrained scalarization.
    """

    def __init__(self, base_problem: ProfiledMultimodalProblem,
                 penalty_scale: float = DEFAULT_ALGO_SWEEP.moead_penalty):
        super().__init__(
            n_var=base_problem.n_var, n_obj=base_problem.n_obj,
            xl=base_problem.xl, xu=base_problem.xu,
            vtype=base_problem.vtype,
        )
        self.base_problem = base_problem
        self.penalty_scale = float(penalty_scale)
        self.profile = base_problem.profile
        self.extras = base_problem.extras
        self.scenario = base_problem.scenario

    def _evaluate(self, X, out, *args, **kwargs):
        inner: Dict[str, np.ndarray] = {}
        self.base_problem._evaluate(X, inner, *args, **kwargs)
        F = np.asarray(inner["F"], dtype=float)
        G = np.asarray(inner.get("G", np.zeros((len(F), 1))), dtype=float)
        out["F"] = F + np.maximum(G, 0).sum(axis=1, keepdims=True) * self.penalty_scale


# --------------------------------------------------------------------------
# Instrumentation
# --------------------------------------------------------------------------

class MetricsCallback(Callback):
    """Per-generation hypervolume, IGD, spacing and feasibility ratio.

    This callback is what Section 7.3 calls *instrumentation*; disabling it
    roughly halves the wall-clock time of a run.
    """

    def __init__(self, reference_front=None, reference_point=None, enabled: bool = True):
        super().__init__()
        self.reference_front = reference_front
        self.reference_point = reference_point
        self.enabled = enabled
        self.data["history"] = []

    def notify(self, algorithm):
        if not self.enabled:
            return
        pop = algorithm.pop
        
        if isinstance(algorithm.problem, PenaltyProblem):
            X = pop.get("X")
            base = algorithm.problem.base_problem
            F_orig, G_orig, _ = base._evaluator(X, base.profile, base.extras, base.scenario)
            F = np.asarray(F_orig, dtype=float)
            G = np.asarray(G_orig, dtype=float)
        else:
            F = pop.get("F")
            G = pop.get("G")

        if G is not None and np.ndim(G) > 1:
            feasible = np.all(G <= 0, axis=1)
        elif G is not None:
            feasible = np.asarray(G <= 0).ravel()
        else:
            feasible = np.ones(len(F), dtype=bool)
        feasible_F = F[feasible]

        hv = np.nan
        igd = np.nan
        spacing = np.nan
        
        if feasible_F.size:
            if self.reference_point is not None:
                hv = float(HV(ref_point=self.reference_point)(feasible_F))
            if self.reference_front is not None:
                igd = float(IGD(self.reference_front)(feasible_F))
            spacing = compute_spacing(feasible_F)

        self.data["history"].append({
            "generation": algorithm.n_gen,
            "hypervolume": hv,
            "igd": igd,
            "spacing": spacing,
            "feasible_ratio": float(np.mean(feasible)),
            "n_feasible": int(np.sum(feasible)),
            "population_size": int(len(pop)),
        })


def compute_spacing(F: np.ndarray) -> float:
    if F is None or len(F) < 2:
        return np.nan
    d = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
    d[d == 0] = np.inf
    nearest = d.min(axis=1)
    return float(np.std(nearest, ddof=1))


# --------------------------------------------------------------------------
# Algorithm construction
# --------------------------------------------------------------------------

def make_operators(problem: Problem, mode: str, crossover_prob: float,
                   crossover_eta: float, mutation_eta: float,
                   path_mutation_prob: float) -> Dict[str, object]:
    """Return the sampling/crossover/mutation triple for the encoding in use."""
    if mode == "path":
        from src.network.operators import (
            MultimodalIndex, PathCrossover, PathMutation, PathSampling,
        )
        G = getattr(problem, "graph", None) or getattr(problem.scenario, "G", None)
        if G is None:
            raise ValueError(
                "path encoding requires a multimodal graph; attach it to the "
                "problem (problem.graph) or to the scenario (scenario.G)"
            )
        index = getattr(problem, "graph_index", None) or MultimodalIndex(G)
        problem.graph_index = index
        return dict(
            sampling=PathSampling(G, index=index),
            crossover=PathCrossover(G, prob=crossover_prob),
            mutation=PathMutation(G, index=index, prob=path_mutation_prob),
        )
    return dict(
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=crossover_prob, eta=crossover_eta),
        mutation=PM(eta=mutation_eta),
    )


def make_algorithm(
    name: str,
    problem: Problem,
    plan: str,
    n_partitions: int,
    crossover_prob: float = DEFAULT_EXPERIMENT.crossover_prob,
    crossover_eta: float = DEFAULT_EXPERIMENT.crossover_eta,
    mutation_eta: float = DEFAULT_EXPERIMENT.mutation_eta,
    stabilized_weights: Optional[Sequence[float]] = None,
    raw_weights: Optional[Sequence[float]] = None,
    rho: float = DEFAULT_REFDIRS.rho,
    encoding: str = "path",
    path_mutation_prob: float = DEFAULT_EXPERIMENT.path_mutation_prob,
):
    """Instantiate the pymoo algorithm for ``name`` under ``plan``."""
    algo = canonical_algorithm(name)
    common = make_operators(problem, encoding, crossover_prob, crossover_eta,
                            mutation_eta, path_mutation_prob)
    # Object-encoded variables cannot be compared by pymoo's default numeric
    # duplicate filter.
    if encoding == "path":
        from src.network.operators import PathDuplicateElimination
        common["eliminate_duplicates"] = PathDuplicateElimination()

    ref_dirs = None
    if algo in REF_DIR_ALGORITHMS:
        ref_dirs = build_reference_directions(
            algo, problem.n_obj, n_partitions,
            stabilized_weights=stabilized_weights,
            raw_weights=raw_weights,
            rho=rho,
        )

    pop_size = resolve_population_size(
        algo, plan, len(ref_dirs) if ref_dirs is not None else None
    )

    if algo == "nsga2":
        return NSGA2(pop_size=pop_size, **common)

    if algo in ("pi_nsga3", "pi_nsga3_raw", "pi_nsga3_stab", "canonical_nsga3"):
        # pymoo silently raises pop_size to len(ref_dirs); passing it
        # explicitly keeps the effective size unambiguous and auditable.
        return NSGA3(pop_size=pop_size, ref_dirs=ref_dirs, **common)

    if algo == "moead":
        common.pop("eliminate_duplicates", None)
        return MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=min(DEFAULT_ALGO_SWEEP.moead_neighbors, len(ref_dirs)),
            **common,
        )

    if algo == "smsemoa":
        return SMSEMOA(pop_size=pop_size, **common)

    raise ValueError(f"Unsupported algorithm: {name}")


# --------------------------------------------------------------------------
# Single run
# --------------------------------------------------------------------------

def run_single_algorithm(
    problem: ProfiledMultimodalProblem,
    algorithm_name: str,
    seed: int,
    n_generations: int,
    plan: str,
    n_partitions: int,
    crossover_prob: float = DEFAULT_EXPERIMENT.crossover_prob,
    crossover_eta: float = DEFAULT_EXPERIMENT.crossover_eta,
    mutation_eta: float = DEFAULT_EXPERIMENT.mutation_eta,
    reference_front: Optional[np.ndarray] = None,
    reference_point: Optional[np.ndarray] = None,
    stabilized_weights: Optional[Sequence[float]] = None,
    raw_weights: Optional[Sequence[float]] = None,
    rho: float = DEFAULT_REFDIRS.rho,
    encoding: str = "path",
    instrumented: bool = True,
) -> AlgorithmRunOutput:
    algo = canonical_algorithm(algorithm_name)
    actual = PenaltyProblem(problem) if algo == "moead" and problem.has_constraints() else problem

    algorithm = make_algorithm(
        algo, problem=problem, plan=plan, n_partitions=n_partitions,
        crossover_prob=crossover_prob, crossover_eta=crossover_eta,
        mutation_eta=mutation_eta, stabilized_weights=stabilized_weights,
        raw_weights=raw_weights, rho=rho, encoding=encoding,
    )
    callback = MetricsCallback(reference_front, reference_point, enabled=instrumented)

    start = time.perf_counter()
    result = minimize(actual, algorithm, ("n_gen", n_generations), seed=seed,
                      callback=callback, verbose=False, save_history=False)
    runtime = time.perf_counter() - start

    X = np.asarray(result.pop.get("X"), dtype=object)
    
    if isinstance(actual, PenaltyProblem):
        F_orig, G_orig, _ = problem._evaluator(X, problem.profile, problem.extras, problem.scenario)
        F = np.asarray(F_orig, dtype=float)
        G = np.asarray(G_orig, dtype=float)
    else:
        F = np.asarray(result.pop.get("F"), dtype=float)
        G = result.pop.get("G")
        G = np.zeros((len(F), 1)) if G is None else np.asarray(G, dtype=float)
        
    feasible = np.all(G <= 0, axis=1) if G.ndim > 1 else np.asarray(G <= 0).ravel()

    final_df = pd.DataFrame(index=range(len(F)))
    if encoding == "path":
        final_df["route_nodes"] = ["|".join(X[i, 0].nodes) for i in range(len(F))]
        final_df["route_modes"] = ["|".join(X[i, 0].modes) for i in range(len(F))]
        final_df["n_edges"] = [X[i, 0].n_edges for i in range(len(F))]
    else:
        for i in range(problem.n_var):
            final_df[f"x{i}"] = X[:, i].astype(float)
    for j in range(problem.n_obj):
        final_df[f"obj_{j + 1}"] = F[:, j]
    final_df["constraint_violation"] = G.sum(axis=1) if G.ndim > 1 else G
    final_df["feasible"] = feasible
    final_df["profile_id"] = problem.profile.get("profile_id", "unknown")
    final_df["algorithm"] = algo
    final_df["seed"] = seed
    final_df["runtime_sec"] = runtime

    history_df = pd.DataFrame(callback.data["history"])
    if len(history_df):
        history_df["profile_id"] = problem.profile.get("profile_id", "unknown")
        history_df["algorithm"] = algo
        history_df["seed"] = seed
        history_df["runtime_sec"] = runtime

    return AlgorithmRunOutput(
        str(problem.profile.get("profile_id", "unknown")), algo, seed, runtime,
        final_df, history_df,
    )


# --------------------------------------------------------------------------
# Parallel suite
# --------------------------------------------------------------------------

def _task_paths(output_dir: Path, profile_id: str, algorithm: str, seed: int) -> Tuple[Path, Path]:
    pop = output_dir / "checkpoints" / "population" / f"{profile_id}__{algorithm}__seed{seed}.csv"
    hist = output_dir / "checkpoints" / "history" / f"{profile_id}__{algorithm}__seed{seed}.csv"
    pop.parent.mkdir(parents=True, exist_ok=True)
    hist.parent.mkdir(parents=True, exist_ok=True)
    return pop, hist


def _worker_task(profile, algorithm_name, seed, problem_factory, scenario, output_dir,
                 n_generations, plan, n_partitions, crossover_prob, crossover_eta,
                 mutation_eta, stabilized_weights, raw_weights, rho,
                 reference_front_factory, reference_point_factory, encoding, instrumented):
    profile_id = str(profile.get("profile_id", "unknown"))
    algo = canonical_algorithm(algorithm_name)
    pop_ckpt, hist_ckpt = _task_paths(Path(output_dir), profile_id, algo, seed)

    if pop_ckpt.exists() and hist_ckpt.exists():
        return profile_id, algo, seed, str(pop_ckpt), str(hist_ckpt), True

    problem = problem_factory(profile, scenario, seed)
    output = run_single_algorithm(
        problem=problem, algorithm_name=algo, seed=seed,
        n_generations=n_generations, plan=plan, n_partitions=n_partitions,
        crossover_prob=crossover_prob, crossover_eta=crossover_eta,
        mutation_eta=mutation_eta,
        reference_front=reference_front_factory(profile) if reference_front_factory else None,
        reference_point=reference_point_factory(profile) if reference_point_factory else None,
        stabilized_weights=stabilized_weights, raw_weights=raw_weights, rho=rho,
        encoding=encoding, instrumented=instrumented,
    )
    output.final_population.to_csv(pop_ckpt, index=False)
    output.history.to_csv(hist_ckpt, index=False)
    return profile_id, algo, seed, str(pop_ckpt), str(hist_ckpt), False


def run_algorithm_suite_parallel3_checkpointed(
    problem_factory,
    profiles: Iterable[Dict[str, object]],
    scenario,
    output_dir: str,
    algorithms: Sequence[str],
    seeds: Sequence[int],
    n_generations: int,
    plan: str,
    n_partitions: int,
    crossover_prob: float = DEFAULT_EXPERIMENT.crossover_prob,
    crossover_eta: float = DEFAULT_EXPERIMENT.crossover_eta,
    mutation_eta: float = DEFAULT_EXPERIMENT.mutation_eta,
    stabilized_weights: Optional[Sequence[float]] = None,
    raw_weights: Optional[Sequence[float]] = None,
    rho: float = DEFAULT_REFDIRS.rho,
    reference_front_factory=None,
    reference_point_factory=None,
    max_workers: int = 3,
    encoding: str = "path",
    instrumented: bool = True,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run ``algorithms x profiles x seeds``, resuming from existing checkpoints."""
    output_path = Path(output_dir)
    profiles = list(profiles)
    tasks = [(p, a, s) for p in profiles for s in seeds for a in algorithms]
    lock = Lock()
    population_frames: List[pd.DataFrame] = []
    history_frames: List[pd.DataFrame] = []

    pbar = tqdm(total=len(tasks), desc=f"{plan}", unit="run", disable=not show_progress)

    def store(pop_path: str, hist_path: str) -> None:
        pop_df = pd.read_csv(pop_path)
        hist_df = pd.read_csv(hist_path) if Path(hist_path).stat().st_size > 1 else pd.DataFrame()
        with lock:
            population_frames.append(pop_df)
            if len(hist_df):
                history_frames.append(hist_df)

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="moo") as executor:
        futures = {
            executor.submit(
                _worker_task, profile, algorithm, seed, problem_factory, scenario,
                str(output_path), n_generations, plan, n_partitions,
                crossover_prob, crossover_eta, mutation_eta,
                stabilized_weights, raw_weights, rho,
                reference_front_factory, reference_point_factory,
                encoding, instrumented,
            ): (str(profile.get("profile_id", "unknown")), algorithm, seed)
            for profile, algorithm, seed in tasks
        }
        for future in as_completed(futures):
            profile_id, algorithm, seed = futures[future]
            try:
                _, done_algo, done_seed, pop_path, hist_path, resumed = future.result()
                store(pop_path, hist_path)
                pbar.update(1)
                pbar.set_postfix_str(
                    f"{'resume' if resumed else 'done'} {done_algo} {profile_id} s{done_seed}"
                )
            except Exception as exc:
                pbar.close()
                raise RuntimeError(
                    f"run failed: algo={algorithm}, profile={profile_id}, seed={seed}: {exc}"
                ) from exc

    pbar.close()
    populations = pd.concat(population_frames, ignore_index=True) if population_frames else pd.DataFrame()
    history = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    return populations, history
