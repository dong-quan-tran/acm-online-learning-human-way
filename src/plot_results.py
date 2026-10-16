
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "online_results.csv"
FIGURES_PATH = ROOT / "results" / "figures"


def plot_learning_curve():
    # Read experiment results
    if not RESULTS_PATH.exists():
        print("No online_results.csv file found yet.")
        return

    with open(RESULTS_PATH, "r", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        print("No experiment results found yet.")
        return

    # Group checkpoints by experiment run
    runs = {}

    for row in rows:
        try:
            mode = row["mode"].strip().lower()
            run_id = row["run_id"].strip()
            examples = int(row["examples_processed"])
            accuracy = float(row["accuracy"])
            eval_split = row["eval_split"].strip().lower()

            if not 0 <= accuracy <= 1:
                raise ValueError

        except (ValueError, TypeError, KeyError):
            print("Skipping row with missing or invalid values.")
            continue

        key = (mode, run_id, eval_split)

        if key not in runs:
            runs[key] = []

        runs[key].append((examples, accuracy))

    if not runs:
        print("No valid results to plot.")
        return

    # Create learning curves
    plt.figure(figsize=(9, 5))

    for (mode, run_id, eval_split), points in sorted(runs.items()):
        points.sort(key=lambda point: point[0])

        x = [point[0] for point in points]
        y = [point[1] for point in points]

        plt.plot(
            x, y,
            marker="o",
            label=f"{mode} - {run_id} ({eval_split})"
        )

    plt.xlabel("Training Examples Processed")
    plt.ylabel("Validation Accuracy")
    plt.ylim(0, 1)
    plt.title("Online Learning: Accuracy vs. Training Examples Processed")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the chart
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_PATH / "online_learning_curve.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Learning curve saved to: {output_path}")


if __name__ == "__main__":
    plot_learning_curve()
