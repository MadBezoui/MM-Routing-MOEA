# PI_NSGA3_Core

This directory is a self-contained, modular extraction of the core Multi-Objective Evolutionary Algorithm (MOEA) research pipeline for multimodal transportation routing. It centralizes the models, datasets, execution logic, and generated outputs to facilitate reproduction and understanding of the core methodology.

## Directory Structure

- **`src/`**: Contains the core logic for the simulation, optimization frameworks, and analysis scripts.
  - `pipeline_V6_smart.py`: The main execution orchestrator. It manages demographic stratification, priority weight stabilization, MOEA execution (NSGA-II, NSGA-III, MOEA/D, SMS-EMOA), and the robust recovery of multi-objective metrics (Hypervolume and IGD).
  - `optimization_framework_parallel3.py`: Defines the `ProfiledMultimodalProblem` supporting high-throughput, multi-threaded algorithm evaluation.
  - `comfort_models.py`: Defines the Multi-Layer Perceptron (MLP) surrogate models and heuristic baselines trained directly on the empirical survey data.
  - `survey_data_loader.py` & `config.py`: Core logic for managing hyperparameter configurations and extracting priority weights from respondent profiles.
  - `analytics_V6.py` & `ToDo.py`: Auxiliary scripts used for post-hoc analytical evaluations and the generation of article-ready visualizations.
- **`data/`**: 
  - `survey_results/`: Contains the base empirical survey datasets. These datasets define respondents' demographic data, priority weights, trip constraints, and baseline comfort tolerance thresholds used for calibration.
- **`results/`**: 
  - `outputs_v5_parallel_3threads_fixed/`: The comprehensive output directory. Includes generation checkpoints across algorithms, final generation populations, and comparative hypervolume datasets for the primary and extended benchmark scenarios.
- **`figures/`**: 
  - High-resolution PDF and PNG artifacts of the experimental results. Highlights include multi-algorithm convergence profiles (`figure6_hypervolume_convergence`) and per-profile hypervolume boxplots (`figure8_per_profile_hv_convergence_variable_small_gap`).

## Key Innovations in Pipeline V6

The core execution logic in `pipeline_V6_smart.py` implements several major robustness improvements critical for the associated publication:
1. **Hamilton Apportionment**: Utilized for balanced, stratified sampling of participant profiles. This ensures proportional demographic and archetype representation in the test subsets.
2. **Weight Stabilization**: The pipeline automatically bounds and stabilizes the empirical priority weights (e.g., `time`, `cost`, `emissions`, `comfort`) to prevent reference-point space collapse (e.g., zero weights) during NSGA-III reference direction targeting.
3. **Robust Feasibility Handling**: Improves hypervolume and IGD recovery by applying safe boolean coercion logic to filter strings and corrupted NaN spaces from the feasibility logs.
4. **Surrogate Comfort Evaluation**: The pipeline leverages an MLP regressor pre-trained on the survey dataset to dynamically evaluate non-linear multimodal routing comfort scores during the genetic generation loop.

## Usage

To re-execute the core optimization module:
```bash
python src/pipeline_V6_smart.py \
    --survey_dir data/survey_results \
    --output_dir results/outputs_v5_parallel_3threads_fixed \
    --max_workers 5
```
