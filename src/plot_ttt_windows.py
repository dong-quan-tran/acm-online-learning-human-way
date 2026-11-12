
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = Path("results/ttt_exp_b_seed42_windows.csv")
OUTPUT_DIR = Path("results/plots")

def main():
    df = pd.read_csv(INPUT_FILE)

    required = [
        "run_id", "experiment", "condition", "seed",
        "window_index", "start_sample", "end_sample",
        "window_size", "correct_count", "accuracy"
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for (experiment, seed), group in df.groupby(
        ["experiment", "seed"]
    ):
        for corruption in ["clean", "gaussian_noise"]:
            subset = group[
                group["run_id"].str.contains(
                    f"_{corruption}_seed", regex=False
                )
            ]

            if subset.empty:
                continue

            plt.figure(figsize=(9, 5))

            for condition in [
                "joint_frozen", "standard_ttt", "online_ttt"
            ]:
                method = subset[
                    subset["condition"] == condition
                ].sort_values("window_index")

                if not method.empty:
                    plt.plot(
                        method["window_index"] + 1,
                        method["accuracy"] * 100,
                        marker="o",
                        label=condition
                    )

            plt.xlabel("Window (200 images each)")
            plt.ylabel("Accuracy (%)")
            plt.title(
                f"Experiment {experiment} | Seed {seed} | {corruption}"
            )
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()

            output = OUTPUT_DIR / (
                f"exp_{experiment.lower()}_seed{seed}_{corruption}.png"
            )

            plt.savefig(output, dpi=200)
            plt.close()

            print(f"Saved: {output}")

if __name__ == "__main__":
    main()
