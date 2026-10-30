import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = Path(
    os.environ.get("RESULTS_CSV", str(ROOT / "results" / "online_results.csv"))
)
FIGURES_PATH = ROOT / "results" / "figures"


def plot_learning_curve():
    if not RESULTS_PATH.exists():
        print("No online_results.csv file found yet.")
        return

    with open(RESULTS_PATH, "r", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        print("No experiment results found yet.")
        return

    runs = {}
    required = ("mode", "run_id", "examples_processed", "accuracy", "eval_split")
    for row in rows:
        try:
            if any(not isinstance(row.get(key), str) or not row[key].strip() for key in required):
                raise ValueError
            mode = row["mode"].strip().lower()
            run_id = row["run_id"].strip()
            examples = int(row["examples_processed"])
            accuracy = float(row["accuracy"])
            eval_split = row["eval_split"].strip().lower()
            if examples < 0 or not 0 <= accuracy <= 1:
                raise ValueError
        except (ValueError, TypeError, KeyError):
            print("Skipping row with missing or invalid values.")
            continue
        runs.setdefault((mode, run_id, eval_split), []).append((examples, accuracy))

    if not runs:
        print("No valid results to plot.")
        return

    plt.figure(figsize=(9, 5))
    for (mode, run_id, eval_split), points in sorted(runs.items()):
        points.sort(key=lambda point: point[0])
        plt.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label=f"{mode} - {run_id} ({eval_split})",
        )
    plt.xlabel("Training Examples Processed")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Online Learning: Accuracy vs. Training Examples Processed")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)   
    output_path = FIGURES_PATH / os.environ.get("FIGURE_NAME", "online_learning_curve.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Learning curve saved to: {output_path}")


if __name__ == "__main__":
    plot_learning_curve()
