# MoRoute-MOEA

**MoRoute-MOEA** is an advanced Python framework designed for solving Multimodal Transportation Routing problems using Multi-Objective Evolutionary Algorithms (MOEAs).

The framework models the city network as a highly detailed integrated graph combining both OpenStreetMap (OSM) pedestrian networks and General Transit Feed Specification (GTFS) public transportation schedules. It employs a discrete path-based genetic representation and state-of-the-art optimization algorithms (NSGA-II, NSGA-III, SMS-EMOA, and MOEA/D) to discover optimal Pareto fronts balancing multiple conflicting routing objectives.

## Features

- **Integrated Multimodal Graph Engine**: Automatically fuses OSM `.pbf` structures with `.zip` GTFS transit schedules into a singular `NetworkX` graph.
- **Path-Based Routing Operators**: Uses specialized genetic crossover (`PathCrossover`), mutation (`PathMutation`), and sampling operators operating directly on graph structures, ensuring 100% valid generated routes.
- **Multi-Objective Evaluator**: Computes true travel time, precise monetary costs (including subscriptions), emissions (using standard environmental models), and a machine-learning based comfort surrogate model.
- **Comfort Surrogate Pipeline**: A 12-feature Multi-Layer Perceptron (MLP) trained on empirical survey data, evaluated using robust `GroupShuffleSplit` cross-validation to prevent data leakage and tested against input Gaussian noise for high-fidelity evaluation.
- **Borda-Count Priority Stabilization**: Objectively calibrates the reference directions for many-objective algorithms (like NSGA-III) using smoothed Borda-count matrices extracted from traveler preference surveys.

## Quick Start

### Prerequisites
Make sure you have `osmium-tool`, `conda`, and all required Python packages installed. The environment heavily relies on `networkx`, `pymoo`, `osmnx`, `pandas`, and `scikit-learn`.

### Running the Optimizer
You can launch the complete pipeline via the experiment runner. This script orchestrates the synthetic profile generation, subgraph filtering for rapid routing, and executes the MOEAs seamlessly.

```bash
export PYTHONPATH=.:src
python src/experiment_runner.py --graph-path data/processed/strasbourg_multimodal.graphml --out-dir outputs_smart_routing
```

### Data Pipeline
To construct the integrated graph from raw files (`strasbourg.osm.pbf` and `strasbourg_gtfs.zip`), run the multimodal network builder:
```bash
export PYTHONPATH=.:src
python src/network/builder.py
```

## Authors
This repository is dedicated to continuous research in AI-driven smart city mobility and multiobjective transportation routing.
