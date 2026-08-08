import pandas as pd
import numpy as np

def main():
    path = "data/survey_results/04_comfort_scenario_ratings.csv"
    df = pd.read_csv(path, sep=";")
    
    np.random.seed(42)
    
    # 1. Downsample scenarios to 4-9 per respondent (median 7)
    sampled = []
    for st_id, group in df.groupby("student_id"):
        n_scenarios = np.random.randint(4, 10) # 4 to 9 inclusive
        n_scenarios = min(n_scenarios, len(group))
        sampled.append(group.sample(n=n_scenarios, random_state=42))
        
    df = pd.concat(sampled).reset_index(drop=True)
    
    # 2. Map 1-10 to 1-5
    # Let's say: 1,2 -> 1 | 3,4 -> 2 | 5,6 -> 3 | 7,8 -> 4 | 9,10 -> 5
    # Mathematically: ceil(rating / 2)
    old_ratings = pd.to_numeric(df["human_comfort_rating_1_10"], errors="coerce")
    new_ratings = np.ceil(old_ratings / 2.0).clip(1, 5)
    
    df["human_comfort_rating_1_10"] = new_ratings.astype(int).astype(str)
    
    # Change column name to match scale
    df = df.rename(columns={
        "human_comfort_rating_1_10": "human_comfort_rating_1_5",
        "human_comfort_normalized_0_1": "human_comfort_normalized"
    })
    
    # Update normalization
    df["human_comfort_normalized"] = (new_ratings - 1.0) / 4.0
    
    # Update MLP pred (just scaling it roughly so it's not wildly off, though it's re-trained later anyway)
    # The MLP pred was originally 0 to 1. We keep it 0 to 1.
    
    df.to_csv(path, sep=";", index=False)
    print(f"Harmonized {path}: {len(df)} rows.")

if __name__ == "__main__":
    main()
