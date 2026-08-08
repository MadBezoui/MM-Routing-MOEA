"""config.py
=============
Centralised, version-controlled hyperparameters.

Every constant quoted in the manuscript appears here exactly once, so that the
released configuration and the reported configuration cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------
# Survey calibration
# --------------------------------------------------------------------------

@dataclass
class SurveyCalibration:
    """Aggregate anchors derived from the 749 respondents (Section 4.1).

    The defaults below are placeholders only.  ``survey_data_loader.load_all``
    always overwrites them with values computed from the raw CSV files.
    """

    sample_size: int = 749
    mean_age: float = 25.8
    mean_distance_to_campus_km: float = 4.86
    mean_daily_budget_eur: float = 5.66
    mean_max_travel_time_min: float = 36.6
    walking_threshold_km: float = 1.16
    weather_importance: float = 4.22 / 5.0
    safety_importance: float = 3.89 / 5.0
    reliability_importance: float = 3.67 / 5.0
    comfort_over_emission_prob: float = 0.60
    comfort_over_cost_prob: float = 0.68
    eco_attitude_mean: float = 3.55 / 5.0


# --------------------------------------------------------------------------
# Comfort surrogate (Section 4.1)
# --------------------------------------------------------------------------

#: The twelve components of phi(P).  Section 4.1: "the five mode shares,
#: crowding, transfers, distance, weather indicator, temperature, age, and
#: mobility restriction".
COMFORT_FEATURES: Tuple[str, ...] = (
    "walk_share", "bike_share", "bus_share", "tram_share", "car_share",
    "crowding", "transfers", "distance_km", "rain", "temperature_c",
    "age", "mobility_restriction",
)


@dataclass
class ComfortTrainingConfig:
    """Architecture and training protocol of the multilayer perceptron."""

    #: Respondent-level hold-out fraction (leakage-free split, Section 4.1).
    test_size: float = 0.2
    random_state: int = 42

    hidden_layers: Tuple[int, int] = (100, 50)
    activation: str = "relu"
    #: L2 regularisation, alpha = 1e-2.
    alpha: float = 1e-2
    max_iter: int = 500
    early_stopping: bool = True
    #: Early stopping monitors a 20 % held-out validation split.
    validation_fraction: float = 0.2
    learning_rate_init: float = 1e-3

    #: Standard deviations used by the input-feature noise study (Section 6.6).
    noise_levels: Sequence[float] = (0.0, 0.02, 0.05, 0.10)

    feature_columns: Tuple[str, ...] = COMFORT_FEATURES


# --------------------------------------------------------------------------
# Priority-weight stabilization (Section 4.2)
# --------------------------------------------------------------------------

@dataclass
class StabilizationConfig:
    """Parameters of Eq. 6 and the admissibility set of Eq. 7."""

    #: Uniform-blend intensity beta.
    blend_uniform: float = 0.20
    #: Per-component floor phi.  ``phi_min`` is the smallest admissible floor.
    floor: float = 0.08
    floor_min: float = 0.08
    #: Calibration grid used by the (beta, phi) sweep of Section 6.6.
    beta_grid: Sequence[float] = (0.0, 0.10, 0.20, 0.30)
    phi_grid: Sequence[float] = (0.0, 0.04, 0.08)


# --------------------------------------------------------------------------
# Reference directions (Section 4.3)
# --------------------------------------------------------------------------

@dataclass
class ReferenceDirectionConfig:
    """Das-Dennis lattice plus priority-informed anchors."""

    #: Number of divisions p of the Das-Dennis lattice.
    divisions_main: int = 8
    divisions_extended: int = 7
    #: Anchor-spread parameter rho of Eq. 9.
    rho: float = 0.30
    #: Sweep used by the sensitivity analysis of Section 6.6 (Table 13).
    rho_grid: Sequence[float] = (0.10, 0.20, 0.30, 0.50)
    #: Directions are deduplicated after rounding to this many decimals.
    dedup_decimals: int = 6


# --------------------------------------------------------------------------
# Experimental plans (Section 5.3)
# --------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Global experimental settings shared by the four plans."""

    total_profiles_expected: int = 150
    representative_subset_size: int = 10
    extended_subset_size: int = 30
    ablation_subset_size: int = 30

    #: Simulated binary crossover and polynomial mutation settings.  The
    #: per-variable mutation probability p_m = 1/n is pymoo's default.
    crossover_prob: float = 0.9
    crossover_eta: float = 15.0
    mutation_eta: float = 20.0
    #: Per-route mutation probability of the path operator (Section 4.3).
    path_mutation_prob: float = 0.2

    profile_id_col: str = "profile_id"
    archetype_col: str = "archetype"
    trip_bin_col: str = "trip_distance_bin"


#: Number of generations per plan.
GENERATIONS: Dict[str, int] = {
    "main": 150,
    "convergence": 150,
    "extended": 120,
    "ablation": 150,
    "equalization": 150,
    "sensitivity": 100,
}

#: Number of random seeds per plan and algorithm (Section 5.3).
SEEDS: Dict[str, Dict[str, int]] = {
    "main": {"nsga2": 30, "pi_nsga3": 30},
    "convergence": {"nsga2": 30, "pi_nsga3": 30},
    "extended": {"nsga2": 10, "pi_nsga3": 10, "moead": 10, "smsemoa": 5},
    "ablation": {"nsga2": 10, "canonical_nsga3": 10, "pi_nsga3_raw": 10, "pi_nsga3_stab": 10},
}

#: Das-Dennis divisions per plan.
DIVISIONS: Dict[str, int] = {
    "main": 8, "convergence": 8, "extended": 7,
    "ablation": 8, "equalization": 8, "sensitivity": 8,
}


@dataclass
class AlgorithmSweepConfig:
    algorithms: Sequence[str] = ("nsga2", "pi_nsga3", "moead", "smsemoa")
    moead_neighbors: int = 20
    #: Penalty coefficient of the MOEA/D Tchebycheff scalarization (Table 4).
    moead_penalty: float = 1e3


# --------------------------------------------------------------------------
# Normalization and evaluation
# --------------------------------------------------------------------------

@dataclass
class NormalizationConfig:
    """Hypervolume normalization schemes compared in Section 6.6."""

    default_scheme: str = "union_observed_max_per_profile"
    schemes: Sequence[str] = (
        "union_observed_max_per_profile",   # Eq. 12-13, algorithm-agnostic
        "per_algorithm_max",
        "fixed_survey_nadir",
        "ideal_bounded",
    )
    epsilon: float = 1e-6


@dataclass
class ScenarioConfig:
    """Stochastic multipliers of Eq. 2 and Eq. 3.

    All flags are ``False`` in the main deterministic experiments, so that
    ``pi = omega = 1``.  They are switched on only by the Monte Carlo study of
    Section 6.6.
    """

    dynamic_pricing: bool = False
    stochastic_travel_time: bool = False
    stochastic_crowding: bool = False
    congestion_multiplier_mean: float = 1.0
    congestion_multiplier_std: float = 0.10
    occupancy_multiplier_mean: float = 1.0
    occupancy_multiplier_std: float = 0.15
    pricing_multiplier_mean: float = 1.0
    pricing_multiplier_std: float = 0.12
    n_monte_carlo: int = 1
    comfort_bias: float = 0.0
    G: Optional[Any] = None
    origins: Optional[List[str]] = None
    destinations: Optional[List[str]] = None


@dataclass
class BenchmarkConfig:
    use_paired_statistics: bool = True
    compute_intrinsic_dimensionality: bool = True
    pca_components: int = 2
    wilcoxon_zero_method: str = "wilcox"
    tie_tolerance: float = 1e-6
    #: Profile-stratified bootstrap resamples for the CI of Table 8.
    bootstrap_resamples: int = 10000
    #: Family-wise error control applied to the 150 per-profile tests.
    multiplicity_correction: str = "holm"
    export_folder: str = "outputs"


DEFAULT_SURVEY = SurveyCalibration()
DEFAULT_COMFORT_CONFIG = ComfortTrainingConfig()
DEFAULT_STABILIZATION = StabilizationConfig()
DEFAULT_REFDIRS = ReferenceDirectionConfig()
DEFAULT_EXPERIMENT = ExperimentConfig()
DEFAULT_ALGO_SWEEP = AlgorithmSweepConfig()
DEFAULT_NORMALIZATION = NormalizationConfig()
DEFAULT_SCENARIO = ScenarioConfig()
DEFAULT_BENCHMARK = BenchmarkConfig()


# --------------------------------------------------------------------------
# Population sizes (Section 5.3)
# --------------------------------------------------------------------------

#: Explicit population size per (plan, algorithm).  For the reference-direction
#: methods the value equals the cardinality of the reference set, so that every
#: direction is associated with at least one individual; it is therefore fixed
#: by the construction of Section 4.3 rather than tuned.
POPULATION_SIZES: Dict[str, Dict[str, int]] = {
    "smoke": {
        "nsga2": 20,
        "nsga3": 20,
        "pi_nsga3": 20,
    },
    "main": {
        "nsga2": 168,
        "pi_nsga3": 170, "pi_nsga3_raw": 170, "pi_nsga3_stab": 170,
        "canonical_nsga3": 165,
        "smsemoa": 168, "moead": 165,
    },
    "convergence": {
        "nsga2": 168,
        "pi_nsga3": 170, "pi_nsga3_raw": 170, "pi_nsga3_stab": 170,
        "canonical_nsga3": 165,
        "smsemoa": 168, "moead": 165,
    },
    # Extended benchmark: a single budget N = 128 with p = 7.
    "extended": {
        "nsga2": 128, "smsemoa": 128,
        "pi_nsga3": 128, "pi_nsga3_raw": 128, "pi_nsga3_stab": 128,
        "canonical_nsga3": 128, "moead": 128,
    },
    "ablation": {
        "nsga2": 168,
        "canonical_nsga3": 165,
        "pi_nsga3_raw": 170, "pi_nsga3_stab": 170, "pi_nsga3": 170,
    },
    "equalization": {
        "nsga2": 170, "pi_nsga3": 170, "pi_nsga3_stab": 170, "pi_nsga3_raw": 170,
    },
    "sensitivity": {
        "nsga2": 168,
        "pi_nsga3": 170, "pi_nsga3_raw": 170, "pi_nsga3_stab": 170,
        "canonical_nsga3": 165,
    },
}

#: Canonical algorithm names.  ``nsga3`` is accepted as a legacy alias of the
#: proposed variant so that result files produced before the rename keep
#: loading.
ALGORITHM_ALIASES: Dict[str, str] = {
    "nsga3": "pi_nsga3",
    "nsga3_informed": "pi_nsga3",
    "pi-nsga-iii": "pi_nsga3",
    "nsga-ii": "nsga2",
    "sms-emoa": "smsemoa",
    "moea/d": "moead",
}


def canonical_algorithm(name: str) -> str:
    """Normalise an algorithm label to its canonical form."""
    key = str(name).strip().lower()
    return ALGORITHM_ALIASES.get(key, key)


def resolve_population_size(
    algorithm: str,
    plan: str,
    n_reference_directions: Optional[int] = None,
) -> int:
    """Return the explicit population size for ``(algorithm, plan)``.

    Raises rather than silently falling back, so that an unconfigured
    combination cannot enter an experiment unnoticed.
    """
    algo = canonical_algorithm(algorithm)
    plan_key = str(plan).strip().lower()

    table = POPULATION_SIZES.get(plan_key)
    if table is not None and algo in table:
        return table[algo]

    if n_reference_directions is not None:
        return int(n_reference_directions)

    raise ValueError(
        f"No population size configured for algorithm '{algorithm}' in plan "
        f"'{plan}'. Add it to config.POPULATION_SIZES."
    )
