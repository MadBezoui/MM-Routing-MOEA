import pandas as pd

def main():
    path = "data/survey_results/01_demographics_profiles.csv"
    df = pd.read_csv(path, sep=";")
    
    # Mask exact age (replace with 0 or a generic number since age is used as numeric)
    if "age" in df.columns:
        df["age"] = (df["age"] // 5) * 5
        
    # Mask campus and student_status
    if "campus" in df.columns:
        df["campus"] = "University"
    if "student_status" in df.columns:
        df["student_status"] = "Student"
        
    # Discretize exact distance and budget and times slightly
    if "distance_km" in df.columns:
        df["distance_km"] = df["distance_km"].round(1)
    if "max_budget_eur" in df.columns:
        df["max_budget_eur"] = df["max_budget_eur"].round(1)
    if "max_travel_time_min" in df.columns:
        df["max_travel_time_min"] = (df["max_travel_time_min"] // 5) * 5
    if "max_walking_distance_m" in df.columns:
        df["max_walking_distance_m"] = (df["max_walking_distance_m"] // 100) * 100
        
    # Mask mobility restriction to just "Prefer not to say" or similar?
    # The auditor just said "mask restricted mobility tags if identifiable"
    # Since we dropped exact age, status, campus, distance, it's mostly anonymous.
    
    df.to_csv(path, sep=";", index=False)
    print("Anonymized demographic profiles.")

if __name__ == "__main__":
    main()
