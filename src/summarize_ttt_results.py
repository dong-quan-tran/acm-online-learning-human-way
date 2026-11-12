
from pathlib import Path
import sys
import pandas as pd

# Default to real Experiment B results.
RESULTS_FILE = Path(
    sys.argv[1] if len(sys.argv) > 1
    else "results/ttt_exp_b_seed42.csv"
)

OUTPUT_FILE = Path("results/ttt_multiseed_summary.csv")

def main():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {RESULTS_FILE}")

    df = pd.read_csv(RESULTS_FILE)

    required = [
        "experiment", "condition", "seed", "checkpoint",
        "dataset", "split", "corruption", "severity",
        "accuracy", "runtime_seconds", "update_count"
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["corruption"] = df["corruption"].fillna("clean")
    df["severity"] = df["severity"].fillna("none").astype(str)

    # Each adaptive run must match a frozen run from
    # the same experiment, seed, and checkpoint.
    match_cols = [
        "experiment", "seed", "checkpoint", "dataset",
        "split", "corruption", "severity"
    ]

    baseline = df[df["condition"] == "joint_frozen"].copy()
    adaptive = df[
        df["condition"].isin(["standard_ttt", "online_ttt"])
    ].copy()

    if baseline.duplicated(match_cols).any():
        raise ValueError("Duplicate frozen baseline found.")

    if adaptive.duplicated(match_cols + ["condition"]).any():
        raise ValueError("Duplicate adaptive run found.")

    baseline = baseline[
        match_cols + ["accuracy"]
    ].rename(columns={"accuracy": "baseline_accuracy"})

    compared = adaptive.merge(
        baseline,
        on=match_cols,
        how="left",
        validate="many_to_one"
    )

    if compared["baseline_accuracy"].isna().any():
        raise ValueError("Adaptive run missing matched baseline.")

    compared["accuracy_change_pp"] = (
        compared["accuracy"] - compared["baseline_accuracy"]
    ) * 100

    group_cols = [
        "experiment", "dataset", "split",
        "condition", "corruption", "severity"
    ]

    summary = compared.groupby(group_cols).agg(
        seeds=("seed", "nunique"),
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        mean_change_pp=("accuracy_change_pp", "mean"),
        std_change_pp=("accuracy_change_pp", "std"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
        mean_updates=("update_count", "mean")
    ).reset_index()

    summary = summary.sort_values(group_cols)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_FILE, index=False)

    print("\n=== TTT Multi-Seed Summary ===")
    print(summary.to_string(index=False))

    print(f"\nSaved: {OUTPUT_FILE}")

    if (summary["seeds"] < 3).any():
        print(
            "\nNote: Some groups have fewer than 3 seeds. "
            "Results are preliminary."
        )

if __name__ == "__main__":
    main()
