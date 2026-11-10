
from pathlib import Path
import pandas as pd

RESULTS_FILE = Path("results/ttt_results.csv")

def main():
    if not RESULTS_FILE.exists():
        print("No TTT results yet. Waiting for experiment runs.")
        return

    df = pd.read_csv(RESULTS_FILE)

    required = [
        "experiment", "condition", "seed", "checkpoint",
        "dataset", "split", "corruption", "severity",
        "accuracy", "runtime_seconds", "update_count"
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Use explicit labels for clean data and missing severity.
    df["corruption"] = df["corruption"].fillna("clean")
    df["severity"] = df["severity"].fillna("none").astype(str)

    # Match runs on their experimental settings.
    match_cols = [
        "experiment", "seed", "checkpoint", "dataset",
        "split", "corruption", "severity"
    ]

    baseline = df[df["condition"] == "joint_frozen"].copy()
    adaptive = df[df["condition"] != "joint_frozen"].copy()

    if baseline.duplicated(match_cols).any():
        raise ValueError("Duplicate frozen baseline matches found.")

    baseline = baseline[match_cols + ["accuracy"]].rename(
        columns={"accuracy": "baseline_accuracy"}
    )

    compared = adaptive.merge(
        baseline, on=match_cols, how="left", validate="many_to_one"
    )

    if compared["baseline_accuracy"].isna().any():
        raise ValueError(
            "Some adaptive runs have no matched frozen baseline."
        )

    compared["accuracy_change_pp"] = (
        compared["accuracy"] - compared["baseline_accuracy"]
    ) * 100

    # Summarize experiments separately.
    for experiment, group in compared.groupby("experiment"):
        print(f"\n=== Experiment {experiment} ===")

        summary = group.groupby(
            ["condition", "corruption", "severity"]
        ).agg(
            runs=("accuracy", "count"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_change_pp=("accuracy_change_pp", "mean"),
            std_change_pp=("accuracy_change_pp", "std"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            mean_updates=("update_count", "mean"),
        )

        print(summary.to_string())

if __name__ == "__main__":
    main()
