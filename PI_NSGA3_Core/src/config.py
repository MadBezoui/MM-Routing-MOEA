from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class SurveyCalibration:
    """Empirical survey anchors used to calibrate comfort heuristics.

    These defaults are aligned with the summary you provided and can be replaced
    with campus- or city-specific estimates loaded from CSV files.
    """

    sample_size: int = 749
    mean_age: float = 25.8
    mean_distance_to_campus_km: float = 4.86
    mean_daily_budget_eur: float = 5.66
    walking_threshold_km: float = 1.16
    weather_importance: float = 4.22 / 5.0
    safety_importance: float = 3.89 / 5.0
    reliability_importance: float = 3.67 / 5.0
    comfort_over_emission_prob: float = 0.60
    comfort_over_cost_prob: float = 0.68
    eco_attitude_mean: float = 3.55 / 5.0


@dataclass
class ComfortTrainingConfig:
    n_samples: int = 5000
    test_size: float = 0.2
    random_state: int = 42
    hidden_layers: Tuple[int, int] = (100, 50)
    activation: str = "relu"
    alpha: float = 0.01
    max_iter: int = 500
    early_stopping: bool = True
    learning_rate_init: float = 1e-3
    noise_levels: Sequence[float] = (0.0, 0.02, 0.05, 0.10)
    route_feature_columns: Tuple[str, ...] = (
        "walk_share",
        "bike_share",
        "bus_share",
        "tram_share",
        "car_share",
        "crowding",
        "transfers",
        "distance_km",
        "rain",
        "temperature_c",
        "age",
        "mobility_restriction",
        "reliability_penalty",
        "safety_penalty",
        "fare_eur",
        "travel_time_min",
    )


@dataclass
class ExperimentConfig:
    random_seeds: Sequence[int] = tuple(range(30))
    representative_subset_size: int = 10
    total_profiles_expected: int = 150
    n_generations: int = 150
    population_size: int = 168
    crossover_prob: float = 0.9
    mutation_eta: float = 20.0
    crossover_eta: float = 15.0
    profile_id_col: str = "profile_id"
    archetype_col: str = "archetype"
    trip_bin_col: str = "trip_distance_bin"


@dataclass
class AlgorithmSweepConfig:
    algorithms: Sequence[str] = ("nsga2", "nsga3", "moead", "smsemoa")
    nsga3_divisions: int = 8
    nsga3_informed_reference_directions: Optional[int] = 35
    moead_neighbors: int = 20
    smsemoa_population_size: int = 168


@dataclass
class NormalizationConfig:
    schemes: Sequence[str] = (
        "max_observed",
        "reference_front",
        "fixed_reference_points",
    )
    ideal_point: Optional[Sequence[float]] = None
    nadir_point: Optional[Sequence[float]] = None


@dataclass
class ScenarioConfig:
    dynamic_pricing: bool = False
    stochastic_travel_time: bool = False
    stochastic_crowding: bool = False
    congestion_multiplier_mean: float = 1.0
    congestion_multiplier_std: float = 0.10
    occupancy_multiplier_mean: float = 1.0
    occupancy_multiplier_std: float = 0.15
    pricing_multiplier_mean: float = 1.0
    pricing_multiplier_std: float = 0.12


@dataclass
class BenchmarkConfig:
    use_paired_statistics: bool = True
    compute_intrinsic_dimensionality: bool = True
    tsne_perplexity: int = 20
    pca_components: int = 2
    wilcoxon_zero_method: str = "wilcox"
    tie_tolerance: float = 1e-6
    export_folder: str = "outputs"


DEFAULT_SURVEY = SurveyCalibration()
DEFAULT_COMFORT_CONFIG = ComfortTrainingConfig()
DEFAULT_EXPERIMENT = ExperimentConfig()
DEFAULT_ALGO_SWEEP = AlgorithmSweepConfig()
DEFAULT_NORMALIZATION = NormalizationConfig()
DEFAULT_SCENARIO = ScenarioConfig()
DEFAULT_BENCHMARK = BenchmarkConfig()
