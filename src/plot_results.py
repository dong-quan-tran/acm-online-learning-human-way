
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "results.csv"
FIGURES_PATH = ROOT / "results" / "figures"


def plot_baseline():
    # Read experiment results
    with open(RESULTS_PATH, "r", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    # Find the baseline experiment
    baseline_rows = [
        row for row in rows
        if row.get("method", "").strip().lower() == "baseline"
    ]

    if not baseline_rows:
        print("No baseline results found yet.")
        return

    result = baseline_rows[0]

    # Get the three required accuracy measurements
    columns = [
        "task_1_acc_after_task_1",
        "task_1_acc_after_task_2",
        "task_2_acc_after_task_2",
    ]

    try:
        accuracies = [float(result[column]) for column in columns]
    except (ValueError, TypeError, KeyError):
        print("Baseline accuracy values are missing or invalid.")
        return

    if not all(0 <= accuracy <= 1 for accuracy in accuracies):
        print("Expected accuracy values as decimals between 0 and 1.")
        return

    labels = [
        "Task 1 after Task 1",
        "Task 1 after Task 2",
        "Task 2 after Task 2",
    ]

    # Create the baseline accuracy chart
    plt.figure(figsize=(9, 5))
    plt.bar(labels, accuracies)
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Baseline Accuracy Across Sequential Tasks")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    # Save the chart
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_PATH / "baseline_forgetting.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Baseline plot saved to: {output_path}")


if __name__ == "__main__":
    plot_baseline()
