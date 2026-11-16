"""Generate publication-quality stream accuracy plot for Experiment B.

Visualizes rolling window accuracy across test images for evaluated conditions
on Clean and Gaussian Noise streams directly from ``ttt_window_summary.csv``.
"""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_stream_plot(
    window_csv_path: str = "results/ttt_window_summary.csv",
    output_png_path: str = "results/stream_accuracy_plot.png",
):
    window_path = Path(window_csv_path)
    if not window_path.exists():
        raise FileNotFoundError(f"Missing {window_csv_path}")

    # Read window CSV
    data_by_run = {}
    with open(window_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row["run_id"]
            if run_id not in data_by_run:
                data_by_run[run_id] = {
                    "condition": row["condition"],
                    "seed": int(row["seed"]),
                    "windows": [],
                }
            data_by_run[run_id]["windows"].append({
                "window_index": int(row["window_index"]),
                "start_sample": int(row["start_sample"]),
                "end_sample": int(row["end_sample"]),
                "window_size": int(row["window_size"]),
                "correct_count": int(row["correct_count"]),
                "accuracy": float(row["accuracy"]),
            })

    if not data_by_run:
        raise ValueError(f"No window rows found in {window_csv_path}")

    # Sort windows by index
    for run_info in data_by_run.values():
        run_info["windows"].sort(key=lambda w: w["window_index"])

    # Group runs by stream
    clean_runs = {k: v for k, v in data_by_run.items() if "clean" in k}
    noise_runs = {
        k: v
        for k, v in data_by_run.items()
        if "gaussian_noise" in k or "noisy" in k
    }

    all_windows = [w for info in data_by_run.values() for w in info["windows"]]
    max_x = max(w["end_sample"] for w in all_windows)
    window_sizes = sorted({w["window_size"] for w in all_windows})
    window_label = f"Window Size = {window_sizes[0]}" if len(window_sizes) == 1 else "Rolling Window"

    all_y = [w["accuracy"] * 100.0 for w in all_windows]
    y_lo = max(0.0, math.floor((min(all_y) - 5.0) / 5.0) * 5.0)
    y_hi = min(100.0, math.ceil((max(all_y) + 8.0) / 5.0) * 5.0)
    if y_hi - y_lo < 10.0:
        y_lo = max(0.0, y_lo - 5.0)
        y_hi = min(100.0, y_hi + 5.0)

    # Style mapping
    style_map = {
        "supervised_frozen": {
            "label": "Supervised Frozen (No TTT)",
            "color": "#7f7f7f",
            "linestyle": ":",
            "marker": "s",
            "linewidth": 1.8,
        },
        "joint_frozen": {
            "label": "Joint Frozen (No TTT)",
            "color": "#1f77b4",
            "linestyle": "--",
            "marker": "o",
            "linewidth": 2.0,
        },
        "standard_ttt": {
            "label": "Standard TTT (Reset per image)",
            "color": "#ff7f0e",
            "linestyle": "-.",
            "marker": "^",
            "linewidth": 2.2,
        },
        "online_ttt": {
            "label": "Online TTT (Carryover state)",
            "color": "#2ca02c",
            "linestyle": "-",
            "marker": "D",
            "linewidth": 2.8,
        },
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Plot Clean Stream
    seen_clean_labels = set()
    for _run_id, run_info in sorted(clean_runs.items()):
        cond = run_info["condition"]
        style = style_map.get(cond, {"label": cond, "color": "black", "linestyle": "-", "marker": "."})
        label = style["label"] if style["label"] not in seen_clean_labels else None
        if label:
            seen_clean_labels.add(label)
        x = [(w["start_sample"] + w["end_sample"]) / 2 for w in run_info["windows"]]
        y = [w["accuracy"] * 100.0 for w in run_info["windows"]]
        ax1.plot(
            x, y,
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=style.get("linewidth", 2),
            markersize=6,
            alpha=0.9,
        )

    ax1.set_title("Stream 1: Clean Images (No Corruption)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Stream Sample Position (Test Image Index)", fontsize=11)
    ax1.set_ylabel(f"Window Accuracy (%) [{window_label}]", fontsize=11)
    ax1.set_xlim(0, max_x)
    ax1.set_ylim(y_lo, y_hi)
    ax1.grid(True, linestyle="--", alpha=0.5)
    if seen_clean_labels:
        ax1.legend(loc="lower left", framealpha=0.95, fontsize=10)

    # Plot Gaussian Noise Stream
    seen_noise_labels = set()
    for _run_id, run_info in sorted(noise_runs.items()):
        cond = run_info["condition"]
        style = style_map.get(cond, {"label": cond, "color": "black", "linestyle": "-", "marker": "."})
        label = style["label"] if style["label"] not in seen_noise_labels else None
        if label:
            seen_noise_labels.add(label)
        x = [(w["start_sample"] + w["end_sample"]) / 2 for w in run_info["windows"]]
        y = [w["accuracy"] * 100.0 for w in run_info["windows"]]
        ax2.plot(
            x, y,
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=style.get("linewidth", 2),
            markersize=6,
            alpha=0.9,
        )

    ax2.set_title("Stream 2: Gaussian Noise (σ = 0.08, Severity 3)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Stream Sample Position (Test Image Index)", fontsize=11)
    ax2.set_xlim(0, max_x)
    ax2.set_ylim(y_lo, y_hi)
    ax2.grid(True, linestyle="--", alpha=0.5)
    if seen_noise_labels:
        ax2.legend(loc="upper left", framealpha=0.95, fontsize=10)

    # Compute Online TTT vs Joint Frozen delta directly from the CSV rows
    online_noise = [v for v in noise_runs.values() if v["condition"] == "online_ttt"]
    joint_noise = [v for v in noise_runs.values() if v["condition"] == "joint_frozen"]
    if online_noise and joint_noise:
        def _weighted_acc_pct(runs):
            total_correct = sum(w["correct_count"] for r in runs for w in r["windows"])
            total_samples = sum(w["window_size"] for r in runs for w in r["windows"])
            return (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0

        delta_pp = _weighted_acc_pct(online_noise) - _weighted_acc_pct(joint_noise)
        peak_w = max(
            (w for r in online_noise for w in r["windows"]),
            key=lambda w: w["accuracy"],
        )
        peak_x = (peak_w["start_sample"] + peak_w["end_sample"]) / 2.0
        peak_y = peak_w["accuracy"] * 100.0
        y_span = max(y_hi - y_lo, 1.0)
        text_x = (
            min(max_x * 0.65, peak_x + max_x * 0.08)
            if peak_x < max_x * 0.65
            else max(max_x * 0.08, peak_x - max_x * 0.35)
        )
        text_y = (
            min(y_hi - y_span * 0.08, peak_y + y_span * 0.16)
            if peak_y + y_span * 0.20 <= y_hi
            else max(y_lo + y_span * 0.08, peak_y - y_span * 0.18)
        )
        summary_word = "gain" if delta_pp >= 0 else "change"
        ax2.annotate(
            f"Online TTT: {delta_pp:+.1f} pp {summary_word}\nvs. Joint Frozen",
            xy=(peak_x, peak_y),
            xytext=(text_x, text_y),
            arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=8),
            bbox=dict(boxstyle="round,pad=0.5", fc="#eafaf1", ec="#2ca02c", lw=1.5),
            fontsize=10,
            fontweight="bold",
            color="#145a32",
        )

    seeds = sorted({v["seed"] for v in data_by_run.values()})
    seed_label = f"Seed {seeds[0]}" if len(seeds) == 1 else f"Seeds {', '.join(map(str, seeds))}"
    fig.suptitle(
        f"Experiment B: Test-Time Training Stream Dynamics on CIFAR-100 (Classes 0–19, {seed_label})",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    out_p = Path(output_png_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved successfully to {out_p}")


if __name__ == "__main__":
    generate_stream_plot()
