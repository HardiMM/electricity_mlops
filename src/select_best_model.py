import pandas as pd
import shutil
from pathlib import Path

# Paths
results_dir = Path("results")
models_dir = Path("models")

# Read experiment metrics
files = [
    results_dir / "experiment_1_metrics.csv",
    results_dir / "experiment_2_metrics.csv",
    results_dir / "experiment_3_metrics.csv"
]

data = []

for file in files:
    df = pd.read_csv(file)

    experiment_name = file.stem.replace("_metrics", "")

    data.append({
        "Experiment": experiment_name,
        "MAE": df["MAE"].iloc[0],
        "RMSE": df["RMSE"].iloc[0],
        "MAPE": df["MAPE"].iloc[0],
        "R2": df["R2"].iloc[0]
    })

# Create comparison table
comparison = pd.DataFrame(data)

# Sort by RMSE
comparison = comparison.sort_values("RMSE", ascending=True)

# Save comparison
comparison.to_csv(
    results_dir / "model_comparison.csv",
    index=False
)

# Select best experiment
best_experiment = comparison.iloc[0]["Experiment"]

# Copy best model
source_model = models_dir / f"{best_experiment}.keras"
best_model = models_dir / "best_model.keras"

shutil.copy(source_model, best_model)

print("\nModel Performance Comparison:")
print(comparison)

print("\nBest Model:", best_experiment)
print("Best Model saved as:", best_model)