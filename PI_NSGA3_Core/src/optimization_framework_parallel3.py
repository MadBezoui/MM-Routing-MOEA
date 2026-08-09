from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import time

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
from pymoo.util.ref_dirs import get_reference_directions
from tqdm.auto import tqdm

Config.warnings["not_compiled"] = False

ObjectiveEvaluator = Callable[[np.ndarray, Dict[str, object], Dict[str, object], object], Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]]


@dataclass
class AlgorithmRunOutput:
    profile_id: str
    algorithm: str
    seed: int
    runtime_sec: float
    final_population: pd.DataFrame
    history: pd.DataFrame


class ProfiledMultimodalProblem(Problem):
    def __init__(
        self,
        n_var: int,
        n_obj: int,
        xl: Sequence[float],
        xu: Sequence[float],
        evaluator: ObjectiveEvaluator,
        profile: Dict[str, object],
        extras: Dict[str, object],
        scenario: object,
        n_ieq_constr: int = 1,
    ):
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr, xl=np.asarray(xl), xu=np.asarray(xu))
        self._evaluator = evaluator
        self.profile = profile
        self.extras = extras
        self.scenario = scenario
        self.latest_meta: Dict[str, np.ndarray] = {}

    def _evaluate(self, X, out, *args, **kwargs):
        F, G, meta = self._evaluator(X, self.profile, self.extras, self.scenario)
        out["F"] = np.asarray(F, dtype=float)
        out["G"] = np.asarray(G, dtype=float)
        self.latest_meta = meta


class PenaltyProblem(Problem):
    def __init__(self, base_problem: ProfiledMultimodalProblem, penalty_scale: float = 1e3):
        super().__init__(n_var=base_problem.n_var, n_obj=base_problem.n_obj, xl=base_problem.xl, xu=base_problem.xu)
        self.base_problem = base_problem
        self.penalty_scale = penalty_scale
        self.profile = base_problem.profile
        self.extras = base_problem.extras
        self.scenario = base_problem.scenario

    def _evaluate(self, X, out, *args, **kwargs):
        inner = {}
        self.base_problem._evaluate(X, inner, *args, **kwargs)
        F = np.asarray(inner["F"], dtype=float)
        G = np.asarray(inner.get("G", np.zeros((len(F), 1))), dtype=float)
        penalty = np.maximum(G, 0).sum(axis=1, keepdims=True) * self.penalty_scale
        out["F"] = F + penalty


class MetricsCallback(Callback):
    def __init__(self, reference_front: Optional[np.ndarray] = None, reference_point: Optional[np.ndarray] = None):
        super().__init__()
        self.reference_front = reference_front
        self.reference_point = reference_point
        self.data["history"] = []

    def notify(self, algorithm):
        pop = algorithm.pop
        F = pop.get("F")
        G = pop.get("G")
        feasible = np.all(G <= 0, axis=1) if G is not None and len(np.shape(G)) > 1 else (G <= 0 if G is not None else np.ones(len(F), dtype=bool))
        feasible_F = F[feasible] if feasible.any() else F

        hv = np.nan
        if self.reference_point is not None and feasible_F.size:
            hv = float(HV(ref_point=self.reference_point)(feasible_F))

        igd = np.nan
        if self.reference_front is not None and feasible_F.size:
            igd = float(IGD(self.reference_front)(feasible_F))

        spacing = compute_spacing(feasible_F) if feasible_F.size else np.nan
        self.data["history"].append(
            {
                "generation": algorithm.n_gen,
                "hypervolume": hv,
                "igd": igd,
                "spacing": spacing,
                "feasible_ratio": float(np.mean(feasible)),
                "n_feasible": int(np.sum(feasible)),
                "population_size": int(len(pop)),
            }
        )


def compute_spacing(F: np.ndarray) -> float:
    if F is None or len(F) < 2:
        return np.nan
    d = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
    d[d == 0] = np.inf
    nearest = d.min(axis=1)
    return float(np.std(nearest, ddof=1)) if len(nearest) > 1 else 0.0


def weighted_reference_directions(n_obj: int, n_partitions: int, priority_weights: Optional[Sequence[float]] = None) -> np.ndarray:
    base = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
    if priority_weights is None:
        return base
    w = np.asarray(priority_weights, dtype=float)
    if w.ndim != 1 or len(w) != n_obj:
        return base
    w = np.clip(w, 1e-9, None)
    w = w / w.sum()
    anchors = [w]
    eye = np.eye(n_obj)
    for e in eye:
        anchors.append(0.7 * w + 0.3 * e)
    anchors = np.asarray(anchors)
    anchors = anchors / anchors.sum(axis=1, keepdims=True)
    merged = np.vstack([base, anchors])
    return np.unique(np.round(merged, 6), axis=0)


def make_algorithm(name: str, n_obj: int, population_size: int, n_partitions: int, crossover_prob: float, crossover_eta: float, mutation_eta: float, priority_weights: Optional[Sequence[float]] = None):
    common = dict(
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=crossover_prob, eta=crossover_eta),
        mutation=PM(eta=mutation_eta),
    )
    ref_dirs = weighted_reference_directions(n_obj, n_partitions, priority_weights=priority_weights)
    lname = name.lower()
    if lname == "nsga2":
        return NSGA2(pop_size=population_size, **common)
    if lname == "nsga3":
        return NSGA3(pop_size=max(population_size, len(ref_dirs)), ref_dirs=ref_dirs, **common)
    if lname == "moead":
        return MOEAD(ref_dirs=ref_dirs, n_neighbors=min(20, len(ref_dirs)), **common)
    if lname == "smsemoa":
        return SMSEMOA(pop_size=population_size, **common)
    raise ValueError(f"Unsupported algorithm: {name}")


def run_single_algorithm(
    problem: ProfiledMultimodalProblem,
    algorithm_name: str,
    seed: int,
    n_generations: int,
    population_size: int,
    n_partitions: int,
    crossover_prob: float,
    crossover_eta: float,
    mutation_eta: float,
    reference_front: Optional[np.ndarray] = None,
    reference_point: Optional[np.ndarray] = None,
    priority_weights: Optional[Sequence[float]] = None,
) -> AlgorithmRunOutput:
    actual_problem = PenaltyProblem(problem) if algorithm_name.lower() == "moead" and problem.has_constraints() else problem
    algorithm = make_algorithm(
        algorithm_name,
        problem.n_obj,
        population_size=population_size,
        n_partitions=n_partitions,
        crossover_prob=crossover_prob,
        crossover_eta=crossover_eta,
        mutation_eta=mutation_eta,
        priority_weights=priority_weights,
    )
    callback = MetricsCallback(reference_front=reference_front, reference_point=reference_point)
    start = time.perf_counter()
    result = minimize(actual_problem, algorithm, ("n_gen", n_generations), seed=seed, callback=callback, verbose=False, save_history=False)
    runtime = time.perf_counter() - start

    X = np.asarray(result.pop.get("X"))
    F = np.asarray(result.pop.get("F"))
    G = result.pop.get("G")
    if G is None:
        G = np.zeros((len(F), 1))
    G = np.asarray(G)
    feasible = np.all(G <= 0, axis=1) if G.ndim > 1 else (G <= 0)

    final_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(problem.n_var)])
    for j in range(problem.n_obj):
        final_df[f"obj_{j+1}"] = F[:, j]
    final_df["constraint_violation"] = G.sum(axis=1) if G.ndim > 1 else G
    final_df["feasible"] = feasible
    final_df["profile_id"] = problem.profile.get("profile_id", "unknown")
    final_df["algorithm"] = algorithm_name.lower()
    final_df["seed"] = seed
    final_df["runtime_sec"] = runtime

    history_df = pd.DataFrame(callback.data["history"])
    history_df["profile_id"] = problem.profile.get("profile_id", "unknown")
    history_df["algorithm"] = algorithm_name.lower()
    history_df["seed"] = seed
    history_df["runtime_sec"] = runtime
    return AlgorithmRunOutput(str(problem.profile.get("profile_id", "unknown")), algorithm_name.lower(), seed, runtime, final_df, history_df)


def _task_paths(output_dir: Path, profile_id: str, algorithm_name: str, seed: int) -> Tuple[Path, Path]:
    pop_path = output_dir / "checkpoints" / "population" / f"{profile_id}__{algorithm_name}__seed{seed}.csv"
    hist_path = output_dir / "checkpoints" / "history" / f"{profile_id}__{algorithm_name}__seed{seed}.csv"
    pop_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    return pop_path, hist_path


def _worker_task(
    profile: Dict[str, object],
    algorithm_name: str,
    seed: int,
    problem_factory: Callable[[Dict[str, object], object], ProfiledMultimodalProblem],
    scenario: object,
    output_dir: str,
    n_generations: int,
    population_size: int,
    n_partitions: int,
    crossover_prob: float,
    crossover_eta: float,
    mutation_eta: float,
    priority_weights: Optional[Sequence[float]],
    reference_front_factory: Optional[Callable[[Dict[str, object]], np.ndarray]],
    reference_point_factory: Optional[Callable[[Dict[str, object]], np.ndarray]],
) -> Tuple[str, str, int, str, str, bool]:
    profile_id = str(profile.get("profile_id", "unknown"))
    output_path = Path(output_dir)
    pop_ckpt, hist_ckpt = _task_paths(output_path, profile_id, algorithm_name, seed)

    if pop_ckpt.exists() and hist_ckpt.exists():
        return profile_id, algorithm_name, seed, str(pop_ckpt), str(hist_ckpt), True

    problem = problem_factory(profile, scenario)
    output = run_single_algorithm(
        problem=problem,
        algorithm_name=algorithm_name,
        seed=seed,
        n_generations=n_generations,
        population_size=population_size,
        n_partitions=n_partitions,
        crossover_prob=crossover_prob,
        crossover_eta=crossover_eta,
        mutation_eta=mutation_eta,
        reference_front=reference_front_factory(profile) if reference_front_factory else None,
        reference_point=reference_point_factory(profile) if reference_point_factory else None,
        priority_weights=priority_weights,
    )
    output.final_population.to_csv(pop_ckpt, index=False)
    output.history.to_csv(hist_ckpt, index=False)
    return profile_id, algorithm_name, seed, str(pop_ckpt), str(hist_ckpt), False


def run_algorithm_suite_parallel3_checkpointed(
    problem_factory: Callable[[Dict[str, object], object], ProfiledMultimodalProblem],
    profiles: Iterable[Dict[str, object]],
    scenario: object,
    output_dir: str,
    algorithms: Sequence[str],
    seeds: Sequence[int],
    n_generations: int,
    population_size: int,
    n_partitions: int,
    crossover_prob: float,
    crossover_eta: float,
    mutation_eta: float,
    priority_weights: Optional[Sequence[float]] = None,
    reference_front_factory: Optional[Callable[[Dict[str, object]], np.ndarray]] = None,
    reference_point_factory: Optional[Callable[[Dict[str, object]], np.ndarray]] = None,
    max_workers: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    output_path = Path(output_dir)
    profiles = list(profiles)
    tasks = [(profile, algo, seed) for profile in profiles for seed in seeds for algo in algorithms]
    total = len(tasks)
    lock = Lock()

    population_frames: List[pd.DataFrame] = []
    history_frames: List[pd.DataFrame] = []

    pbar = tqdm(total=total, desc="Optimization-3threads", unit="run", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")

    def read_and_store(pop_path: str, hist_path: str):
        pop_df = pd.read_csv(pop_path)
        hist_df = pd.read_csv(hist_path)
        with lock:
            population_frames.append(pop_df)
            history_frames.append(hist_df)

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="moo") as executor:
        future_map = {
            executor.submit(
                _worker_task,
                profile,
                algorithm_name,
                seed,
                problem_factory,
                scenario,
                str(output_path),
                n_generations,
                population_size,
                n_partitions,
                crossover_prob,
                crossover_eta,
                mutation_eta,
                priority_weights,
                reference_front_factory,
                reference_point_factory,
            ): (str(profile.get("profile_id", "unknown")), algorithm_name, seed)
            for profile, algorithm_name, seed in tasks
        }

        for future in as_completed(future_map):
            profile_id, algorithm_name, seed = future_map[future]
            try:
                done_profile_id, done_algorithm, done_seed, pop_path, hist_path, resumed = future.result()
                read_and_store(pop_path, hist_path)
                status = "resume" if resumed else "done"
                pbar.update(1)
                pbar.set_postfix_str(f"{status} algo={done_algorithm}, profile={done_profile_id}, seed={done_seed}, workers={max_workers}")
            except Exception as e:
                pbar.close()
                raise RuntimeError(f"Parallel task failed for algo={algorithm_name}, profile={profile_id}, seed={seed}: {e}") from e

    pbar.close()
    populations_df = pd.concat(population_frames, ignore_index=True) if population_frames else pd.DataFrame()
    history_df = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    return populations_df, history_df
