"""Generate publication-quality stream accuracy plot for Experiment B.

Visualizes rolling window accuracy across test images for evaluated conditions
on Clean and Gaussian Noise streams directly from ``ttt_window_summary.csv``.
"""

import argparse
import csv
import math
from pathlib import Path
import statistics

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


def generate_held_out_plot(
    held_out_csv_path: str = "results/exp_b_held_out_seed42.csv",
    output_png_path: str = "results/exp_b_held_out_accuracy_plot.png",
) -> None:
    """Plot periodic held-out accuracy trajectories directly from a held-out CSV."""
    csv_path = Path(held_out_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {held_out_csv_path}")

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {held_out_csv_path}")

    required = {
        "condition",
        "seed",
        "adaptation_stream_condition",
        "held_out_condition",
        "adaptation_images_processed",
        "held_out_image_count",
        "held_out_accuracy",
    }
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Held-out CSV missing required column(s): {sorted(missing)}")

    by_view: dict[str, list[dict[str, str]]] = {"clean": [], "noisy": []}
    for row in rows:
        view = row["held_out_condition"]
        if view in by_view:
            by_view[view].append(row)

    all_x = [int(r["adaptation_images_processed"]) for r in rows]
    max_x = max(all_x) if all_x else 0
    x_hi = max(max_x, 1)
    all_y = [float(r["held_out_accuracy"]) * 100.0 for r in rows]
    y_lo = max(0.0, math.floor((min(all_y) - 5.0) / 5.0) * 5.0)
    y_hi = min(100.0, math.ceil((max(all_y) + 8.0) / 5.0) * 5.0)
    if y_hi - y_lo < 10.0:
        y_lo = max(0.0, y_lo - 5.0)
        y_hi = min(100.0, y_hi + 5.0)

    ho_counts = sorted({int(r["held_out_image_count"]) for r in rows})
    count_label = f"{ho_counts[0]} Images" if len(ho_counts) == 1 else "Held-Out Images"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    panel_specs = [
        (ax1, "clean", f"Fixed Clean Held-Out Set ({count_label})"),
        (ax2, "noisy", f"Fixed Gaussian-Noise Held-Out Set (\u03c3 = 0.08, {count_label})"),
    ]
    frozen_styles = {
        "supervised_frozen": {
            "label": "Supervised Frozen (Ref)",
            "color": "#7f7f7f",
            "linestyle": ":",
            "linewidth": 1.8,
        },
        "joint_frozen": {
            "label": "Joint Frozen (Ref)",
            "color": "#1f77b4",
            "linestyle": "--",
            "linewidth": 2.0,
        },
    }
    online_styles = {
        "clean": {
            "label": "Online TTT (Clean Stream)",
            "color": "#ff7f0e",
            "linestyle": "-",
            "marker": "o",
            "linewidth": 2.4,
        },
        "noisy": {
            "label": "Online TTT (Noisy Stream)",
            "color": "#2ca02c",
            "linestyle": "-",
            "marker": "D",
            "linewidth": 2.6,
        },
    }

    def _mean_std(values: list[float]) -> tuple[float, float]:
        mean_val = sum(values) / len(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        return mean_val, std_val

    for ax, view_name, title in panel_specs:
        view_rows = by_view[view_name]
        for cond, style in frozen_styles.items():
            matched = [float(r["held_out_accuracy"]) * 100.0 for r in view_rows if r["condition"] == cond]
            if matched:
                ref_y, ref_std = _mean_std(matched)
                ref_label = (
                    f"{style['label']} ({ref_y:.1f}%)"
                    if len(matched) == 1
                    else f"{style['label']} ({ref_y:.1f}±{ref_std:.1f}%)"
                )
                ax.plot(
                    [0, x_hi],
                    [ref_y, ref_y],
                    label=ref_label,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    alpha=0.9,
                )

        for stream_cond, style in online_styles.items():
            grouped: dict[int, list[float]] = {}
            for r in view_rows:
                if (
                    r["condition"] == "online_ttt"
                    and r["adaptation_stream_condition"] == stream_cond
                ):
                    step = int(r["adaptation_images_processed"])
                    grouped.setdefault(step, []).append(float(r["held_out_accuracy"]) * 100.0)
            if grouped:
                xs = sorted(grouped.keys())
                means = []
                stds = []
                for step in xs:
                    m, s = _mean_std(grouped[step])
                    means.append(m)
                    stds.append(s)
                ax.plot(
                    xs,
                    means,
                    label=style["label"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    linewidth=style["linewidth"],
                    markersize=5.5,
                    alpha=0.95,
                )
                if any(s > 0.0 for s in stds):
                    lo = [max(0.0, m - s) for m, s in zip(means, stds)]
                    hi = [min(100.0, m + s) for m, s in zip(means, stds)]
                    ax.fill_between(xs, lo, hi, color=style["color"], alpha=0.15)

        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Adaptation Images Processed", fontsize=11)
        ax.set_xlim(0, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", framealpha=0.95, fontsize=9.5)

    ax1.set_ylabel("Held-Out Classification Accuracy (%)", fontsize=11)

    noisy_view_rows = by_view["noisy"]
    noisy_grouped: dict[int, list[float]] = {}
    for r in noisy_view_rows:
        if r["condition"] == "online_ttt" and r["adaptation_stream_condition"] == "noisy":
            step = int(r["adaptation_images_processed"])
            noisy_grouped.setdefault(step, []).append(float(r["held_out_accuracy"]) * 100.0)
    joint_noisy = [
        float(r["held_out_accuracy"]) * 100.0
        for r in noisy_view_rows
        if r["condition"] == "joint_frozen"
    ]
    if noisy_grouped and joint_noisy:
        jf_y, _ = _mean_std(joint_noisy)
        mean_by_step = {step: _mean_std(vals)[0] for step, vals in noisy_grouped.items()}
        peak_step = max(mean_by_step.keys(), key=lambda s: mean_by_step[s])
        peak_x = float(peak_step)
        peak_y = mean_by_step[peak_step]
        delta_peak = peak_y - jf_y
        y_span = max(y_hi - y_lo, 1.0)
        text_x = max(x_hi * 0.08, min(x_hi * 0.58, peak_x - x_hi * 0.28))
        text_y = (
            min(y_hi - y_span * 0.12, peak_y + y_span * 0.12)
            if peak_y + y_span * 0.16 <= y_hi
            else max(y_lo + y_span * 0.10, peak_y - y_span * 0.18)
        )
        ax2.annotate(
            f"Peak Noisy Stream: {peak_y:.1f}%\n({delta_peak:+.1f} pp vs. Joint Frozen)",
            xy=(peak_x, peak_y),
            xytext=(text_x, text_y),
            arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=8),
            bbox=dict(boxstyle="round,pad=0.45", fc="#eafaf1", ec="#2ca02c", lw=1.5),
            fontsize=9.5,
            fontweight="bold",
            color="#145a32",
        )

    seeds = sorted({int(r["seed"]) for r in rows})
    seed_label = f"Seed {seeds[0]}" if len(seeds) == 1 else f"Seeds {', '.join(map(str, seeds))}"
    fig.suptitle(
        f"Experiment B: Fixed Held-Out Accuracy During Online TTT (CIFAR-100 Classes 0\u201319, {seed_label})",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    out_p = Path(output_png_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Held-out plot saved successfully to {out_p}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate Experiment B stream or periodic held-out accuracy plots from CSV."
    )
    parser.add_argument(
        "--held-out",
        action="store_true",
        help="Generate periodic held-out accuracy plot instead of legacy rolling-window stream plot",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Input CSV path (defaults to results/exp_b_held_out_seed42.csv for --held-out, else results/ttt_window_summary.csv)",
    )
    parser.add_argument(
        "--output-png",
        type=str,
        default=None,
        help="Output PNG path (defaults to results/exp_b_held_out_accuracy_plot.png for --held-out, else results/stream_accuracy_plot.png)",
    )
    args = parser.parse_args(argv)
    if args.held_out:
        generate_held_out_plot(
            held_out_csv_path=args.input_csv or "results/exp_b_held_out_seed42.csv",
            output_png_path=args.output_png or "results/exp_b_held_out_accuracy_plot.png",
        )
    else:
        generate_stream_plot(
            window_csv_path=args.input_csv or "results/ttt_window_summary.csv",
            output_png_path=args.output_png or "results/stream_accuracy_plot.png",
        )


if __name__ == "__main__":
    main()
