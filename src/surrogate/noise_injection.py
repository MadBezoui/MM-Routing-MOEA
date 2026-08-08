import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.survey_data_loader import load_real_survey_calibration, load_comfort_training_data
from src.comfort_models import SurveyInformedComfortFactory
from config import ComfortTrainingConfig

logger = logging.getLogger(__name__)

def run_noise_injection(survey_dir: str, out_dir: str):
    survey_path = Path(survey_dir)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading real survey calibration from {survey_path}")
    real_survey = load_real_survey_calibration(survey_path)
    real_training_df = load_comfort_training_data(survey_path)
    
    # We will just evaluate the MLP surrogate for the noise injection script
    config = ComfortTrainingConfig()
    config.noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    factory = SurveyInformedComfortFactory(config, real_survey)
    
    logger.info("Training baseline comfort models...")
    # Train models
    results = factory.train_models(real_training_df)
    
    # Isolate MLP
    mlp_result = next((r for r in results if r.model_name == "mlp_surrogate"), None)
    if not mlp_result:
        raise ValueError("MLP surrogate not found in trained models!")
        
    logger.info(f"Evaluating noise robustness across {len(config.noise_levels)} noise levels...")
    noise_df = factory.noise_robustness(mlp_result, real_training_df)
    
    # Include the heuristic for comparison
    heuristic_result = next((r for r in results if r.model_name == "heuristic_direct"), None)
    if heuristic_result:
        heur_noise_df = factory.noise_robustness(heuristic_result, real_training_df)
        noise_df = pd.concat([noise_df, heur_noise_df], ignore_index=True)
    
    csv_path = out_dir_path / "noise_robustness_metrics.csv"
    noise_df.to_csv(csv_path, index=False)
    logger.info(f"Noise robustness metrics saved to {csv_path}")
    
    # Generate visualization
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=noise_df, x="noise_level", y="rmse", hue="model_name", marker="o")
    
    plt.title('Comfort Surrogate Robustness to Input Noise')
    plt.xlabel('Gaussian Noise Level (Std Dev)')
    plt.ylabel('RMSE vs True Comfort Score')
    plt.grid(True, alpha=0.3)
    
    plot_path = out_dir_path / "noise_robustness.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Visualization saved to {plot_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_noise_injection("data/survey_results", "outputs_surrogate")
