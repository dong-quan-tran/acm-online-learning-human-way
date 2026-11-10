
from pathlib import Path
import pandas as pd

# Locate the results file from the repository root.
RESULTS_FILE = Path("results/ttt_results.csv")

def main():
    if not RESULTS_FILE.exists():
        print("No TTT results yet. Waiting for experiment runs.")
        return

    df = pd.read_csv(RESULTS_FILE)

    required = [
        "experiment", "condition", "seed",
        "accuracy", "runtime_seconds", "update_count"
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("Loaded", len(df), "experimental results.")

    for experiment, group in df.groupby("experiment"):
        print(f"\nExperiment {experiment}")
        print(group.groupby("condition")["accuracy"].agg(
            ["count", "mean", "std"]
        ))

if __name__ == "__main__":
    main()
