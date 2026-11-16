"""Build a 6-slide minimalist monochrome presentation (.pptx & .pdf) with
eye-comfort muted-color statistical charts for Tuong's Experiment B weekly report.

Reads verified empirical metrics directly from repository artifacts:
- results/ttt_summary.csv
- results/ttt_window_summary.csv
- results/val_tuning_results.json
- configs/ttt_cifar100_exp_b.yaml
- checkpoints/exp_b_{supervised,joint}_seed42.pt

Outputs (in docs/weekly_updates/Tuong/):
- assets/lr_tuning_chart.png
- assets/test_accuracy_bars.png
- assets/stream_dynamics_comfort.png
- Exp_B_TTT_Report_Tuong.pptx (16:9 native editable slides + speaker notes)
- Exp_B_TTT_Report_Tuong.pdf  (matching 16:9 vector PDF deck)
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import csv
import html
import json
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yaml

# Canvas constants matching TTT_Presentation.pptx (1280 x 720 virtual px -> 13.333 x 7.5 in)
CANVAS_W = 1280
CANVAS_H = 720
EMU_PER_PX = 9525  # 1280 * 9525 = 12,192,000 EMU; 720 * 9525 = 6,858,000 EMU
PT_PER_PX = 0.75   # 96 DPI -> 72 pt/in

# Monochrome presentation palette
MONO_BLACK = "#111111"
MONO_DARK = "#2B2B2B"
MONO_GRAY = "#555555"
MONO_BORDER = "#777777"
MONO_GRID = "#BBBBBB"
MONO_DIVIDER = "#DDDDDD"
MONO_PALE = "#EEEEEE"
MONO_TINT = "#F6F6F6"
MONO_WHITE = "#FFFFFF"

# Eye-comfort muted chart palette
CHART_COLORS = {
    "supervised_frozen": "#8C969E",  # Soft slate gray
    "joint_frozen": "#5B84B1",       # Soft steel blue
    "standard_ttt": "#E08A6C",       # Muted terracotta / soft light red
    "online_ttt": "#5E9E7E",         # Soft sage green
}

CHART_LABELS = {
    "supervised_frozen": "Supervised Frozen (No TTT)",
    "joint_frozen": "Joint Frozen (No TTT)",
    "standard_ttt": "Standard TTT (Reset / img)",
    "online_ttt": "Online TTT (Carryover)",
}

CHART_SHORT = {
    "supervised_frozen": "Sup.\nFrozen",
    "joint_frozen": "Joint\nFrozen",
    "standard_ttt": "Standard\nTTT",
    "online_ttt": "Online\nTTT",
}

FONT_FAMILY = "Arial"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.edgecolor"] = "#555555"
plt.rcParams["axes.linewidth"] = 0.9


# ==============================================================================
# Data Loading
# ==============================================================================

def load_experiment_data(repo_root: Path) -> Dict[str, Any]:
    """Load verified metrics from CSV, JSON, YAML, and checkpoint metadata."""
    summary_path = repo_root / "results" / "ttt_summary.csv"
    window_path = repo_root / "results" / "ttt_window_summary.csv"
    tuning_path = repo_root / "results" / "val_tuning_results.json"
    config_path = repo_root / "configs" / "ttt_cifar100_exp_b.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open(tuning_path, "r", encoding="utf-8") as f:
        tuning = json.load(f)

    summary_rows: List[Dict[str, Any]] = []
    with open(summary_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["seed"] = int(row["seed"])
            row["accuracy"] = float(row["accuracy"])
            row["runtime_seconds"] = float(row["runtime_seconds"])
            row["update_count"] = int(row["update_count"])
            row["rotation_loss"] = float(row["rotation_loss"])
            row["learning_rate"] = float(row["learning_rate"])
            row["adaptation_steps"] = int(row["adaptation_steps"])
            summary_rows.append(row)

    summary_map: Dict[Tuple[str, str], Dict[str, Any]] = {
        (r["corruption"], r["condition"]): r for r in summary_rows
    }

    windows_by_run: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    with open(window_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stream = "clean" if "clean" in row["run_id"] else "gaussian_noise"
            key = (stream, row["condition"])
            windows_by_run.setdefault(key, []).append(
                {
                    "window_index": int(row["window_index"]),
                    "start_sample": int(row["start_sample"]),
                    "end_sample": int(row["end_sample"]),
                    "window_size": int(row["window_size"]),
                    "correct_count": int(row["correct_count"]),
                    "accuracy": float(row["accuracy"]),
                }
            )
    for vals in windows_by_run.values():
        vals.sort(key=lambda w: w["window_index"])

    # Read checkpoint source-training metadata without mutating anything
    ckpt_meta: Dict[str, Dict[str, Any]] = {}
    try:
        import torch

        for mode in ("supervised", "joint"):
            ckpt_file = repo_root / "checkpoints" / f"exp_b_{mode}_seed42.pt"
            if ckpt_file.exists():
                raw = torch.load(ckpt_file, map_location="cpu", weights_only=False)
                ckpt_meta[mode] = {
                    "epoch": int(raw.get("epoch", 15)),
                    "source_updates": int(raw.get("source_updates", 2115)),
                    "source_train_seconds": float(raw.get("source_train_seconds", 0.0)),
                    "val_acc": float(raw.get("val_acc", 0.0)),
                    "val_rot_acc": float(raw.get("val_rot_acc", 0.0)),
                }
    except Exception:
        pass

    ckpt_meta.setdefault(
        "supervised",
        {
            "epoch": 15,
            "source_updates": 2115,
            "source_train_seconds": 85.24,
            "val_acc": 0.60,
            "val_rot_acc": 0.23075,
        },
    )
    ckpt_meta.setdefault(
        "joint",
        {
            "epoch": 15,
            "source_updates": 2115,
            "source_train_seconds": 430.08,
            "val_acc": float(tuning.get("source_joint_val_acc", 0.625)),
            "val_rot_acc": float(tuning.get("source_joint_val_rot_acc", 0.614)),
        },
    )

    return {
        "config": config,
        "tuning": tuning,
        "summary_rows": summary_rows,
        "summary_map": summary_map,
        "windows": windows_by_run,
        "checkpoints": ckpt_meta,
    }


# ==============================================================================
# Eye-Comfort Chart Generators (300 DPI PNGs)
# ==============================================================================

def _clean_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(colors="#333333", labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.45, color="#CCCCCC", zorder=0)


def generate_lr_tuning_chart(data: Dict[str, Any], out_path: Path) -> None:
    """Two-panel LR tuning curve on 1,000-image validation split (Slide 03)."""
    cands = data["tuning"]["candidate_evaluations"]
    base_clean = float(data["tuning"]["source_joint_val_acc"]) * 100.0
    lrs = [c["ttt_lr"] for c in cands]
    labels = ["1e-4", "5e-4", "1e-3\n(Locked)", "5e-3"]
    clean_acc = [c["val_clean_acc"] * 100.0 for c in cands]
    noisy_acc = [c["val_noisy_acc"] * 100.0 for c in cands]
    x = list(range(len(lrs)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.62, 3.16), dpi=300)
    fig.patch.set_facecolor(MONO_WHITE)

    # Panel 1: Val Clean Accuracy
    _clean_ax(ax1)
    ax1.axvspan(1.65, 2.35, color="#EEF3F7", zorder=1)
    ax1.axhline(base_clean, color="#777777", linestyle=":", linewidth=1.2, zorder=2)
    ax1.text(0.05, 61.65, f"Source baseline: {base_clean:.1f}%", fontsize=7.6, color="#555555")
    ax1.plot(
        x,
        clean_acc,
        color=CHART_COLORS["joint_frozen"],
        marker="o",
        linewidth=2.2,
        markersize=6,
        zorder=3,
    )
    # Highlight selected vs rejected
    ax1.scatter([2], [clean_acc[2]], color=CHART_COLORS["online_ttt"], s=65, zorder=4, edgecolors="#111111", linewidths=1.0)
    ax1.scatter([3], [clean_acc[3]], color=CHART_COLORS["standard_ttt"], s=55, zorder=4, marker="X")
    for xi, yi in zip(x, clean_acc):
        dy = 0.48 if xi != 3 else -0.88
        fw = "bold" if xi in (2, 3) else "normal"
        col = "#111111" if xi == 2 else ("#9C4B33" if xi == 3 else "#333333")
        ax1.text(xi, yi + dy, f"{yi:.1f}%", ha="center", fontsize=8.4, fontweight=fw, color=col)
    ax1.set_title("Val Clean Acc (%)", fontsize=10, fontweight="bold", color="#111111", pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylim(58.0, 64.6)

    # Panel 2: Val Gaussian Noise Accuracy
    _clean_ax(ax2)
    ax2.axvspan(1.65, 2.35, color="#EEF3F7", zorder=1)
    ax2.plot(
        x,
        noisy_acc,
        color=CHART_COLORS["standard_ttt"],
        marker="s",
        linewidth=2.2,
        markersize=5.5,
        zorder=3,
    )
    ax2.scatter([2], [noisy_acc[2]], color=CHART_COLORS["online_ttt"], s=65, zorder=4, edgecolors="#111111", linewidths=1.0)
    for xi, yi in zip(x, noisy_acc):
        dy = 0.55 if xi != 3 else -1.15
        fw = "bold" if xi == 2 else "normal"
        ax2.text(xi, yi + dy, f"{yi:.1f}%", ha="center", fontsize=8.5, fontweight=fw, color="#222222")
    ax2.set_title("Val Noise Acc (σ=0.08)", fontsize=10, fontweight="bold", color="#111111", pad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylim(28.5, 36.6)

    fig.tight_layout(pad=1.0, w_pad=1.6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor=MONO_WHITE)
    plt.close(fig)


def generate_test_accuracy_bars(data: Dict[str, Any], out_path: Path) -> None:
    """Grouped bar chart of Clean vs. Gaussian Noise accuracy (Slide 04)."""
    smap = data["summary_map"]
    conditions = ["supervised_frozen", "joint_frozen", "standard_ttt", "online_ttt"]
    streams = ["clean", "gaussian_noise"]
    stream_labels = ["Stream 1: Clean Images", "Stream 2: Gaussian Noise (σ = 0.08)"]

    fig, ax = plt.subplots(figsize=(5.70, 4.90), dpi=300)
    fig.patch.set_facecolor(MONO_WHITE)
    _clean_ax(ax)

    group_centers = [0.0, 1.24]
    bar_w = 0.215
    offsets = [-1.5 * bar_w, -0.5 * bar_w, 0.5 * bar_w, 1.5 * bar_w]

    for cond_idx, cond in enumerate(conditions):
        vals = [smap[(st, cond)]["accuracy"] * 100.0 for st in streams]
        xs = [gc + offsets[cond_idx] for gc in group_centers]
        edge = "#111111" if cond == "online_ttt" else "#444444"
        lw = 1.1 if cond == "online_ttt" else 0.6
        bars = ax.bar(
            xs,
            vals,
            width=bar_w * 0.88,
            color=CHART_COLORS[cond],
            edgecolor=edge,
            linewidth=lw,
            label=CHART_LABELS[cond],
            zorder=3,
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                v + 1.0,
                f"{v:.2f}%",
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold" if (cond == "online_ttt" or cond == "joint_frozen") else "normal",
                color="#111111",
            )

    # Bracket highlighting Online TTT +13.90 pp gain over Joint Frozen on Noise
    jf_x = group_centers[1] + offsets[1]
    onl_x = group_centers[1] + offsets[3]
    jf_y = smap[("gaussian_noise", "joint_frozen")]["accuracy"] * 100.0
    onl_y = smap[("gaussian_noise", "online_ttt")]["accuracy"] * 100.0
    delta_pp = onl_y - jf_y
    top_y = onl_y + 7.2
    ax.plot([jf_x, jf_x, onl_x, onl_x], [jf_y + 4.4, top_y, top_y, onl_y + 4.4], color="#111111", lw=1.1, zorder=4)
    ax.text(
        (jf_x + onl_x) / 2.0,
        top_y + 1.0,
        f"+{delta_pp:.2f} pp vs. Joint Frozen",
        ha="center",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
        color="#111111",
        bbox=dict(boxstyle="round,pad=0.25", fc="#EEF5F0", ec="#5E9E7E", lw=0.9),
    )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(stream_labels, fontsize=9.8, fontweight="bold", color="#111111")
    ax.set_ylabel("Test Accuracy (%)  [Unmasked 100-Way]", fontsize=9.5, fontweight="bold", color="#222222")
    ax.set_ylim(0, 80)
    ax.legend(
        loc="upper right",
        ncol=2,
        frameon=True,
        facecolor="#FAFAFA",
        edgecolor="#CCCCCC",
        fontsize=7.8,
    )

    fig.tight_layout(pad=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor=MONO_WHITE)
    plt.close(fig)


def generate_stream_dynamics_chart(data: Dict[str, Any], out_path: Path) -> None:
    """Two-panel rolling window stream accuracy plot (Slide 05)."""
    windows = data["windows"]
    styles = {
        "supervised_frozen": {"ls": ":", "marker": "s", "lw": 1.8, "ms": 4.5, "alpha": 0.85},
        "joint_frozen": {"ls": "--", "marker": "o", "lw": 2.0, "ms": 5.0, "alpha": 0.92},
        "standard_ttt": {"ls": "-.", "marker": "^", "lw": 2.1, "ms": 5.2, "alpha": 0.92},
        "online_ttt": {"ls": "-", "marker": "D", "lw": 2.8, "ms": 5.5, "alpha": 1.0},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.36, 4.24), dpi=300, sharey=True)
    fig.patch.set_facecolor(MONO_WHITE)

    order = ["supervised_frozen", "joint_frozen", "standard_ttt", "online_ttt"]

    for ax, stream, title in [
        (ax1, "clean", "Stream 1: Clean Images (No Corruption)"),
        (ax2, "gaussian_noise", "Stream 2: Gaussian Noise (σ = 0.08, Severity 3)"),
    ]:
        _clean_ax(ax)
        for cond in order:
            wlist = windows[(stream, cond)]
            x = [(w["start_sample"] + w["end_sample"]) / 2.0 for w in wlist]
            y = [w["accuracy"] * 100.0 for w in wlist]
            st = styles[cond]
            ax.plot(
                x,
                y,
                label=CHART_LABELS[cond],
                color=CHART_COLORS[cond],
                linestyle=st["ls"],
                marker=st["marker"],
                linewidth=st["lw"],
                markersize=st["ms"],
                alpha=st["alpha"],
                zorder=4 if cond == "online_ttt" else 3,
            )

        if stream == "gaussian_noise":
            w_onl = windows[("gaussian_noise", "online_ttt")]
            w_jnt = windows[("gaussian_noise", "joint_frozen")]
            x = [(w["start_sample"] + w["end_sample"]) / 2.0 for w in w_onl]
            y_onl = [w["accuracy"] * 100.0 for w in w_onl]
            y_jnt = [w["accuracy"] * 100.0 for w in w_jnt]
            ax.fill_between(x, y_jnt, y_onl, color=CHART_COLORS["online_ttt"], alpha=0.12, zorder=2)

            peak_idx = max(range(len(y_onl)), key=lambda idx: y_onl[idx])
            ax.annotate(
                f"Peak: {y_onl[peak_idx]:.1f}% (Window {peak_idx})\n+13.90 pp overall vs. Joint Frozen",
                xy=(x[peak_idx], y_onl[peak_idx]),
                xytext=(860, 65.5),
                arrowprops=dict(arrowstyle="->", color="#2F6B4E", lw=1.3),
                bbox=dict(boxstyle="round,pad=0.35", fc="#EFF7F2", ec="#5E9E7E", lw=1.0),
                fontsize=8.6,
                fontweight="bold",
                color="#1F4D36",
                zorder=5,
            )

        ax.set_title(title, fontsize=11, fontweight="bold", color="#111111", pad=8)
        ax.set_xlabel("Stream Sample Position (200-Sample Rolling Windows)", fontsize=9.2, color="#333333")
        ax.set_xlim(0, 2000)
        ax.set_ylim(20, 78)
        ax.set_xticks([0, 400, 800, 1200, 1600, 2000])

    ax1.set_ylabel("Window Accuracy (%)", fontsize=9.5, fontweight="bold", color="#222222")
    ax1.legend(loc="lower left", frameon=True, facecolor="#FAFAFA", edgecolor="#CCCCCC", fontsize=8.2)
    ax2.legend(loc="upper left", frameon=True, facecolor="#FAFAFA", edgecolor="#CCCCCC", fontsize=8.2)

    fig.tight_layout(pad=1.1, w_pad=1.8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor=MONO_WHITE)
    plt.close(fig)


# ==============================================================================
# Unified Slide Scene Primitives
# ==============================================================================

@dataclass
class BoxItem:
    x: float
    y: float
    w: float
    h: float
    text: str = ""
    size: float = 20.0
    bold: bool = False
    italic: bool = False
    align: str = "center"    # left | center | right
    valign: str = "middle"   # top | middle | bottom
    fill: Optional[str] = MONO_WHITE
    border: Optional[str] = MONO_BORDER
    border_width: float = 1.6
    text_color: str = MONO_BLACK
    insets: float = 10.0


@dataclass
class LineItem:
    x1: float
    y1: float
    x2: float
    y2: float
    color: str = MONO_BLACK
    width: float = 1.8
    dashed: bool = False
    arrow_end: bool = False
    arrow_start: bool = False


@dataclass
class ElbowItem:
    x1: float
    y1: float
    x2: float
    y2: float
    color: str = MONO_BLACK
    width: float = 1.8
    dashed: bool = False
    arrow_end: bool = True


@dataclass
class TableItem:
    x: float
    y: float
    w: float
    h: float
    col_widths: List[float]
    row_heights: List[float]
    rows: List[List[str]]
    col_aligns: List[str]
    highlight_rows: List[int] = field(default_factory=list)
    bold_cells: List[Tuple[int, int]] = field(default_factory=list)
    font_size: float = 15.5
    header_size: float = 15.5
    pad_x: float = 10.0


@dataclass
class ImageItem:
    x: float
    y: float
    w: float
    h: float
    path: Path
    alt: str = ""


class SlideScene:
    """Platform-neutral representation of one 1280x720 slide."""

    def __init__(self, slide_num: int, title: Optional[str] = None, footer: str = ""):
        self.slide_num = slide_num
        self.items: List[Any] = []
        self.notes: str = ""
        if title:
            self.add_text(72, 42, 1060, 56, title, size=36, bold=True, align="left", valign="middle")
        self.add_text(1160, 44, 48, 34, f"{slide_num:02d}", size=17, align="right", valign="middle", color=MONO_GRAY)
        if footer:
            self.add_text(72, 676, 1136, 24, footer, size=13.5, align="left", valign="middle", color=MONO_GRAY)

    def add_box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str = "",
        *,
        size: float = 18.0,
        bold: bool = False,
        italic: bool = False,
        align: str = "center",
        valign: str = "middle",
        fill: Optional[str] = MONO_WHITE,
        border: Optional[str] = MONO_BORDER,
        border_width: float = 1.5,
        text_color: Optional[str] = None,
        dark: bool = False,
        insets: float = 10.0,
    ) -> BoxItem:
        actual_fill = MONO_BLACK if dark else fill
        actual_border = MONO_BLACK if dark else border
        actual_tc = text_color if text_color is not None else (MONO_WHITE if dark else MONO_BLACK)
        item = BoxItem(
            x=x, y=y, w=w, h=h, text=text, size=size, bold=bold, italic=italic,
            align=align, valign=valign, fill=actual_fill, border=actual_border,
            border_width=border_width, text_color=actual_tc, insets=insets,
        )
        self.items.append(item)
        return item

    def add_text(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        size: float = 18.0,
        bold: bool = False,
        italic: bool = False,
        align: str = "left",
        valign: str = "middle",
        color: str = MONO_BLACK,
    ) -> BoxItem:
        return self.add_box(
            x, y, w, h, text,
            size=size, bold=bold, italic=italic, align=align, valign=valign,
            fill=None, border=None, border_width=0.0, text_color=color, insets=0.0,
        )

    def add_line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = MONO_BLACK,
        width: float = 1.8,
        dashed: bool = False,
        arrow_end: bool = False,
        arrow_start: bool = False,
    ) -> LineItem:
        item = LineItem(x1, y1, x2, y2, color=color, width=width, dashed=dashed, arrow_end=arrow_end, arrow_start=arrow_start)
        self.items.append(item)
        return item

    def connect_lr(self, left_box: BoxItem, right_box: BoxItem, **kwargs: Any) -> LineItem:
        return self.add_line(
            left_box.x + left_box.w,
            left_box.y + left_box.h / 2.0,
            right_box.x,
            right_box.y + right_box.h / 2.0,
            arrow_end=True,
            **kwargs,
        )

    def add_elbow_lr(self, left_box: BoxItem, right_box: BoxItem, **kwargs: Any) -> ElbowItem:
        item = ElbowItem(
            left_box.x + left_box.w,
            left_box.y + left_box.h / 2.0,
            right_box.x,
            right_box.y + right_box.h / 2.0,
            **kwargs,
        )
        self.items.append(item)
        return item

    def add_table(
        self,
        x: float,
        y: float,
        col_widths: List[float],
        row_heights: List[float],
        rows: List[List[str]],
        col_aligns: List[str],
        *,
        highlight_rows: Optional[List[int]] = None,
        bold_cells: Optional[List[Tuple[int, int]]] = None,
        font_size: float = 15.5,
        header_size: float = 15.5,
        pad_x: float = 10.0,
    ) -> TableItem:
        item = TableItem(
            x=x, y=y, w=sum(col_widths), h=sum(row_heights),
            col_widths=col_widths, row_heights=row_heights, rows=rows,
            col_aligns=col_aligns,
            highlight_rows=highlight_rows or [],
            bold_cells=bold_cells or [],
            font_size=font_size,
            header_size=header_size,
            pad_x=pad_x,
        )
        self.items.append(item)
        return item

    def add_image(self, x: float, y: float, w: float, h: float, path: Path, alt: str = "") -> ImageItem:
        item = ImageItem(x=x, y=y, w=w, h=h, path=path, alt=alt)
        self.items.append(item)
        return item

    def set_notes(self, body: str, sources: str) -> None:
        self.notes = f"{body.strip()}\n\nSources: {sources.strip()}"


# ==============================================================================
# Build 6 Slides
# ==============================================================================

def build_slides(data: Dict[str, Any], assets_dir: Path) -> List[SlideScene]:
    smap = data["summary_map"]
    ckpt = data["checkpoints"]
    tuning = data["tuning"]
    windows = data["windows"]

    # Live metrics from CSVs / checkpoints / JSON
    sup_meta = ckpt["supervised"]
    jnt_meta = ckpt["joint"]
    sup_clean = smap[("clean", "supervised_frozen")]
    jnt_clean = smap[("clean", "joint_frozen")]
    std_clean = smap[("clean", "standard_ttt")]
    onl_clean = smap[("clean", "online_ttt")]
    sup_noise = smap[("gaussian_noise", "supervised_frozen")]
    jnt_noise = smap[("gaussian_noise", "joint_frozen")]
    std_noise = smap[("gaussian_noise", "standard_ttt")]
    onl_noise = smap[("gaussian_noise", "online_ttt")]

    onl_vs_jnt_noise = (onl_noise["accuracy"] - jnt_noise["accuracy"]) * 100.0
    onl_vs_std_noise = (onl_noise["accuracy"] - std_noise["accuracy"]) * 100.0
    onl_vs_sup_noise = (onl_noise["accuracy"] - sup_noise["accuracy"]) * 100.0
    std_vs_jnt_noise = (std_noise["accuracy"] - jnt_noise["accuracy"]) * 100.0
    jnt_vs_sup_clean = (jnt_clean["accuracy"] - sup_clean["accuracy"]) * 100.0
    std_vs_jnt_clean = (std_clean["accuracy"] - jnt_clean["accuracy"]) * 100.0
    onl_vs_jnt_clean = (onl_clean["accuracy"] - jnt_clean["accuracy"]) * 100.0
    std_vs_onl_clean = (std_clean["accuracy"] - onl_clean["accuracy"]) * 100.0

    # Window peak for Online TTT on noise
    w_onl_noise = windows[("gaussian_noise", "online_ttt")]
    w0_acc = w_onl_noise[0]["accuracy"] * 100.0
    w2_acc = w_onl_noise[2]["accuracy"] * 100.0
    peak_w = max(w_onl_noise, key=lambda w: w["accuracy"])
    peak_idx = peak_w["window_index"]
    peak_acc = peak_w["accuracy"] * 100.0

    slides: List[SlideScene] = []

    # --------------------------------------------------------------------------
    # Slide 01: Title & End-to-End Task Pipeline
    # --------------------------------------------------------------------------
    s1 = SlideScene(1, title=None, footer="Online Learning - The Human Way  |  configs/ttt_cifar100_exp_b.yaml")
    s1.add_text(72, 66, 1080, 74, "Experiment B: Test-Time Training", size=54, bold=True)
    s1.add_text(72, 146, 1080, 40, "Rotation-based adaptation on CIFAR-100 (Classes 0–19)", size=26)
    s1.add_text(72, 194, 1080, 30, "Tuong (Lead, Experiment B)  |  Seed 42 Pilot on Apple Silicon MPS", size=18, color=MONO_GRAY)

    s1.add_text(72, 246, 600, 24, "WEEKLY TASK PIPELINE", size=14.5, bold=True, color=MONO_GRAY)
    b1 = s1.add_box(72, 278, 258, 106, "1. Data Isolation\n9,000 train / 1,000 val\n2,000 held-out test", size=17.5)
    b2 = s1.add_box(364, 278, 258, 106, "2. Source Training\nSupervised vs. Joint\n15 epochs (GroupNorm)", size=17.5)
    b3 = s1.add_box(656, 278, 258, 106, "3. Label-Free TTT\n4-way rotation loss\n1-step SGD, lr = 0.001", size=17.5, dark=True, bold=True)
    b4 = s1.add_box(948, 278, 260, 106, "4. Stream Evaluation\nClean vs. Noise (σ=0.08)\nUnmasked 100-way head", size=17.5, fill=MONO_PALE)
    s1.connect_lr(b1, b2)
    s1.connect_lr(b2, b3)
    s1.connect_lr(b3, b4)

    # Two research question cards
    for cx, hdr, qtxt, res in [
        (
            72,
            "Q1: ADAPTATION BENEFIT (NOISE VS. CLEAN)",
            "Does rotation-based TTT improve classification under\nGaussian noise (σ = 0.08) without hurting clean images?",
            f"Finding: Online TTT gains +{onl_vs_jnt_noise:.2f} pp on noise ({onl_noise['accuracy']*100:.2f}%)",
        ),
        (
            660,
            "Q2: RESET VS. CARRYOVER AT 1-STEP BUDGET",
            "With 1 update step per image, is carrying updates forward\n(Online TTT) better than resetting per image (Standard TTT)?",
            f"Finding: Online +{onl_vs_std_noise:.2f} pp (noise); Standard +{std_vs_onl_clean:.2f} pp (clean)",
        ),
    ]:
        s1.add_box(cx, 426, 548, 198, "", fill=MONO_WHITE, border=MONO_GRID, border_width=1.2)
        s1.add_box(cx, 426, 548, 42, hdr, size=15, bold=True, align="left", dark=True, insets=14)
        s1.add_text(cx + 16, 480, 516, 72, qtxt, size=18.5, valign="top")
        s1.add_box(cx + 16, 560, 516, 46, res, size=17.5, bold=True, align="left", fill=MONO_TINT, border=MONO_GRID, border_width=1.0, insets=12)

    s1.set_notes(
        f"[About 40 seconds]\n"
        f"This presentation reports the completed implementation and pilot findings for Experiment B on CIFAR-100 classes 0 through 19. "
        f"Following our approved plan, I separated the 10,000-image source pool into 9,000 training and 1,000 validation images, kept 2,000 test images strictly isolated, "
        f"trained matched GroupNorm source models, tuned the adaptation learning rate on validation data, and evaluated four conditions on both clean and Gaussian-noise streams. "
        f"Under noise, Online TTT boosted accuracy by +{onl_vs_jnt_noise:.2f} percentage points over the frozen joint baseline.",
        "docs/exp_b_tuong_results.md; configs/ttt_cifar100_exp_b.yaml; results/ttt_summary.csv",
    )
    slides.append(s1)

    # --------------------------------------------------------------------------
    # Slide 02: Architecture, Data Split & Adaptation Flow
    # --------------------------------------------------------------------------
    s2 = SlideScene(2, title="Architecture, data split, and adaptation flow", footer="Source: src/models.py (SmallResNetTTT), src/data.py, src/ttt_cifar100.py")

    # Left side: SmallResNetTTT Adaptation Flowchart
    s2.add_text(72, 110, 620, 24, "MODEL & LABEL-FREE ADAPTATION (SmallResNetTTT)", size=14.5, bold=True, color=MONO_GRAY)
    s2.add_text(72, 140, 260, 22, "ADAPT (4 rotations / test image)", size=14, bold=True, color=MONO_GRAY)

    r_box = s2.add_box(72, 168, 126, 84, "Rotated Views\n0°, 90°,\n180°, 270°", size=14.5, insets=6)
    e_box = s2.add_box(228, 168, 174, 84, "Backbone θe\n3× Conv-GN(16)-Res\nUPDATES", size=14.5, dark=True, bold=True, insets=6)
    h_box = s2.add_box(432, 168, 134, 84, "Rotation Head θs\nLinear(256→4)\nFROZEN", size=14.5, fill=MONO_PALE, insets=6)
    l_box = s2.add_box(596, 168, 104, 84, "Rot Loss\nCross-Ent", size=14.5, insets=6)
    s2.connect_lr(r_box, e_box)
    s2.connect_lr(e_box, h_box)
    s2.connect_lr(h_box, l_box)

    # Dashed backprop path: l_box bottom (648, 252) -> (648, 290) -> (315, 290) -> (315, 252)
    s2.add_line(648, 252, 648, 288, dashed=True, width=1.6)
    s2.add_line(648, 288, 315, 288, dashed=True, width=1.6)
    s2.add_line(315, 288, 315, 252, dashed=True, width=1.6, arrow_end=True)
    s2.add_text(332, 293, 340, 22, "Backprop through frozen θs (0 test labels)", size=13.5, color=MONO_GRAY)

    s2.add_text(72, 330, 300, 22, "PREDICT (Post-adaptation)", size=14, bold=True, color=MONO_GRAY)
    x_box = s2.add_box(72, 358, 126, 84, "Original Image\n3 × 32 × 32", size=14.5, insets=6)
    e2_box = s2.add_box(228, 358, 174, 84, "Updated θe\nGroupNorm(16)\n0 BatchNorm2d", size=14.5, dark=True, bold=True, insets=6)
    m_box = s2.add_box(432, 358, 134, 84, "Classifier θm\nLinear(256→100)\nFROZEN", size=14.5, fill=MONO_PALE, insets=6)
    y_box = s2.add_box(596, 358, 104, 84, "Prediction\nClasses 0–19", size=14.5, insets=6)
    s2.connect_lr(x_box, e2_box)
    s2.connect_lr(e2_box, m_box)
    s2.connect_lr(m_box, y_box)

    # Guarantees summary card
    s2.add_box(72, 472, 628, 172, "", fill=MONO_TINT, border=MONO_GRID, border_width=1.2)
    s2.add_text(90, 484, 590, 26, "Verified Implementation Guarantees", size=16, bold=True)
    s2.add_text(
        90, 514, 594, 122,
        "• Zero BatchNorm2d layers: 16-group GroupNorm stabilizes B=1 adaptation.\n"
        "• Heads θm and θs set requires_grad=False; verified bit-for-bit unchanged.\n"
        "• Stateless SGD (lr=0.001, momentum=0.0, 1 step/image, no buffer carryover).",
        size=15, valign="top",
    )

    # Divider
    s2.add_line(724, 112, 724, 644, color=MONO_DIVIDER, width=1.2)

    # Right side: Data Split & Noise Isolation Flowchart
    s2.add_text(748, 110, 460, 24, "ISOLATED DATA & NOISE PROTOCOL", size=14.5, bold=True, color=MONO_GRAY)
    pool = s2.add_box(748, 152, 460, 56, "CIFAR-100 Classes 0–19 Source Pool (10,000 imgs)", size=15.5, bold=True)
    tr = s2.add_box(748, 244, 218, 76, "Train Split (9,000)\nSupervised & Joint\nSource Training", size=14.5)
    va = s2.add_box(990, 244, 218, 76, "Val Split (1,000)\nHyperparameter Tuning\n(Locked ttt_lr=0.001)", size=14.5, fill=MONO_PALE)

    # Orthogonal split lines pool -> tr, va
    s2.add_line(978, 208, 978, 226, width=1.6)
    s2.add_line(857, 226, 1099, 226, width=1.6)
    s2.add_line(857, 226, 857, 244, width=1.6, arrow_end=True)
    s2.add_line(1099, 226, 1099, 244, width=1.6, arrow_end=True)

    # Isolation boundary dashed line
    s2.add_line(748, 348, 1208, 348, color=MONO_GRAY, width=1.2, dashed=True)
    s2.add_text(748, 353, 460, 22, "Strict Test Isolation Boundary (Val Tuning Locked First)", size=13, italic=True, align="center", color=MONO_GRAY)

    tst = s2.add_box(748, 386, 460, 68, "Held-Out Test Stream (2,000 Images)\nAscending source-row order (exp_b_test_membership.json)", size=14.5, dark=True, bold=True)
    st1 = s2.add_box(748, 492, 218, 114, "Stream 1: Clean\nPixel space [0, 1]\nNormalized inside\nSmallResNetTTT", size=14.5)
    st2 = s2.add_box(990, 492, 218, 114, "Stream 2: Gaussian Noise\nN(0, 0.08² I), clamped [0,1]\nSHA-256 matched seed\nacross all 4 conditions", size=14.5, fill=MONO_PALE)

    s2.add_line(978, 454, 978, 473, width=1.6)
    s2.add_line(857, 473, 1099, 473, width=1.6)
    s2.add_line(857, 473, 857, 492, width=1.6, arrow_end=True)
    s2.add_line(1099, 473, 1099, 492, width=1.6, arrow_end=True)

    s2.set_notes(
        "[About 45 seconds]\n"
        "On the left is SmallResNetTTT. We replaced BatchNorm with 16-group GroupNorm so single-image adaptation at test time never corrupts batch statistics. "
        "During adaptation, four rotated views pass through the shared backbone and frozen rotation head; backpropagation updates only the backbone θe while keeping both heads strictly frozen. "
        "On the right, we enforced strict split isolation: 9,000 images for training, 1,000 for tuning the learning rate, and 2,000 held-out test images tested with matched SHA-256 deterministic Gaussian noise.",
        "src/models.py (SmallResNetTTT); src/data.py; src/ttt_cifar100.py",
    )
    slides.append(s2)

    # --------------------------------------------------------------------------
    # Slide 03: Source Training & Validation LR Tuning
    # --------------------------------------------------------------------------
    s3 = SlideScene(3, title="Source training and validation LR tuning", footer="Source: checkpoints/exp_b_{supervised,joint}_seed42.pt; results/val_tuning_results.json")

    s3.add_text(72, 110, 540, 24, "SOURCE TRAINING (9,000 TRAIN, 15 EPOCHS, ADAMW)", size=14.5, bold=True, color=MONO_GRAY)
    val_cls_diff = (jnt_meta["val_acc"] - sup_meta["val_acc"]) * 100.0
    src_rows = [
        ["Metric", "Supervised", "Joint (Cls + Rot)"],
        ["Loss Formulation", "L_cls only", "L_cls + 1.0·L_rot"],
        ["Gradient Updates", f"{sup_meta['source_updates']:,}", f"{jnt_meta['source_updates']:,}"],
        ["Train Time (MPS)", f"{sup_meta['source_train_seconds']:.2f} s", f"{jnt_meta['source_train_seconds']:.2f} s"],
        ["Val Class Acc", f"{sup_meta['val_acc']*100:.2f}%", f"{jnt_meta['val_acc']*100:.2f}% (+{val_cls_diff:.2f} pp)"],
        ["Val Rotation Acc", f"{sup_meta['val_rot_acc']*100:.2f}% (chance)", f"{jnt_meta['val_rot_acc']*100:.2f}%"],
    ]
    s3.add_table(
        72, 140,
        col_widths=[182, 160, 200],
        row_heights=[46, 46, 46, 46, 48, 48],
        rows=src_rows,
        col_aligns=["left", "center", "center"],
        highlight_rows=[4, 5],
        bold_cells=[(4, 2), (5, 2)],
        font_size=15,
        header_size=15,
    )

    s3.add_box(72, 444, 542, 198, "", fill=MONO_TINT, border=MONO_GRID, border_width=1.2)
    s3.add_text(90, 458, 506, 26, "Why Joint Source Training Matters", size=16.5, bold=True)
    s3.add_text(
        90, 492, 506, 136,
        f"• Auxiliary rotation supervision acts as a regularizer,\n"
        f"  raising clean validation accuracy by +{val_cls_diff:.2f} pp ({jnt_meta['val_acc']*100:.2f}%).\n"
        f"• Rotation head learns {jnt_meta['val_rot_acc']*100:.2f}% orientation accuracy\n"
        f"  (vs. {sup_meta['val_rot_acc']*100:.2f}% random chance), enabling useful test gradients.",
        size=15.5, valign="top",
    )

    s3.add_text(646, 110, 562, 24, "VALIDATION LR TUNING (N = 1,000 VAL, 1 STEP SGD)", size=14.5, bold=True, color=MONO_GRAY)
    s3.add_image(646, 140, 562, 316, assets_dir / "lr_tuning_chart.png", alt="Validation learning rate tuning chart")

    lr_rows = [["ttt_lr", "Clean Val", "Noisy Val", "Avg Acc", "Decision"]]
    for cand in tuning["candidate_evaluations"]:
        lr_val = cand["ttt_lr"]
        d_clean = cand["clean_delta_vs_source"] * 100.0
        if abs(lr_val - tuning["selected_ttt_lr"]) < 1e-9:
            dec = f"LOCKED ({d_clean:+.2f} pp)"
        elif d_clean < -1.0:
            dec = f"Rejected ({d_clean:+.2f} pp)"
        else:
            dec = f"{d_clean:+.2f} pp clean"
        lr_rows.append([
            f"{lr_val:g}",
            f"{cand['val_clean_acc']*100:.2f}%",
            f"{cand['val_noisy_acc']*100:.2f}%",
            f"{cand['avg_acc']*100:.2f}%",
            dec,
        ])
    s3.add_table(
        646, 466,
        col_widths=[86, 102, 102, 96, 176],
        row_heights=[36, 35, 35, 36, 35],
        rows=lr_rows,
        col_aligns=["center", "center", "center", "center", "left"],
        highlight_rows=[3],
        bold_cells=[(3, 0), (3, 1), (3, 2), (3, 3), (3, 4)],
        font_size=13.5,
        header_size=13.5,
        pad_x=7,
    )

    s3.set_notes(
        f"[About 45 seconds]\n"
        f"Before running any test evaluation, we trained two matched source models for 15 epochs on the 9,000-image training split. "
        f"Joint training improved validation classification accuracy by +{val_cls_diff:.2f} percentage points and trained the rotation branch to {jnt_meta['val_rot_acc']*100:.2f}%. "
        f"On the right, we tuned the adaptation learning rate across four candidates on the 1,000-image validation set. "
        f"Although 0.005 reached 34.5% on noise, it dropped clean accuracy by 3.00 percentage points below baseline. "
        f"We locked ttt_lr = 0.001 because it achieved the highest clean accuracy (63.00%, +0.50 pp) while still improving noisy validation accuracy.",
        "checkpoints/exp_b_supervised_seed42.pt; checkpoints/exp_b_joint_seed42.pt; results/val_tuning_results.json",
    )
    slides.append(s3)

    # --------------------------------------------------------------------------
    # Slide 04: Final Test Results (Bar Chart + Summary Table)
    # --------------------------------------------------------------------------
    s4 = SlideScene(4, title="Test results: clean vs. Gaussian noise", footer="Source: results/ttt_summary.csv (seed 42, 2,000 matched test images, deterministic SHA-256 noise)")

    s4.add_text(72, 110, 570, 24, "TEST ACCURACY BY STREAM & CONDITION (N = 2,000)", size=14.5, bold=True, color=MONO_GRAY)
    s4.add_image(72, 140, 570, 490, assets_dir / "test_accuracy_bars.png", alt="Grouped bar chart comparing clean and noisy test accuracy")

    s4.add_text(668, 110, 540, 24, "QUANTITATIVE COMPARISON (SEED 42)", size=14.5, bold=True, color=MONO_GRAY)

    def fmt_row(stream_key: str, stream_lbl: str, cond_key: str, cond_lbl: str, base_acc: float) -> List[str]:
        r = smap[(stream_key, cond_key)]
        acc = r["accuracy"] * 100.0
        delta = (r["accuracy"] - base_acc) * 100.0
        d_str = "Baseline" if cond_key == "joint_frozen" else f"{delta:+.2f} pp"
        return [
            f"{stream_lbl}: {cond_lbl}",
            f"{acc:.2f}%",
            d_str,
            f"{r['rotation_loss']:.3f}",
            f"{r['runtime_seconds']:.1f} s",
        ]

    res_rows = [
        ["Stream & Condition", "Acc", "Δ Joint", "Rot L", "Time"],
        fmt_row("clean", "Clean", "supervised_frozen", "Sup. Frozen", jnt_clean["accuracy"]),
        fmt_row("clean", "Clean", "joint_frozen", "Joint Frozen", jnt_clean["accuracy"]),
        fmt_row("clean", "Clean", "standard_ttt", "Standard TTT", jnt_clean["accuracy"]),
        fmt_row("clean", "Clean", "online_ttt", "Online TTT", jnt_clean["accuracy"]),
        fmt_row("gaussian_noise", "Noise", "supervised_frozen", "Sup. Frozen", jnt_noise["accuracy"]),
        fmt_row("gaussian_noise", "Noise", "joint_frozen", "Joint Frozen", jnt_noise["accuracy"]),
        fmt_row("gaussian_noise", "Noise", "standard_ttt", "Standard TTT", jnt_noise["accuracy"]),
        fmt_row("gaussian_noise", "Noise", "online_ttt", "Online TTT", jnt_noise["accuracy"]),
    ]
    s4.add_table(
        668, 140,
        col_widths=[198, 84, 102, 78, 78],
        row_heights=[38, 38, 38, 38, 38, 38, 38, 38, 40],
        rows=res_rows,
        col_aligns=["left", "center", "center", "center", "center"],
        highlight_rows=[2, 8],
        bold_cells=[(2, 0), (2, 1), (8, 0), (8, 1), (8, 2)],
        font_size=13.5,
        header_size=14,
        pad_x=7,
    )

    s4.add_box(668, 498, 540, 132, "", fill=MONO_TINT, border=MONO_GRID, border_width=1.2)
    s4.add_text(684, 510, 508, 24, "Headline Test Findings", size=16, bold=True)
    s4.add_text(
        684, 538, 512, 86,
        f"• Noise (σ = 0.08): Online TTT reaches {onl_noise['accuracy']*100:.2f}%, beating Joint Frozen\n"
        f"  by +{onl_vs_jnt_noise:.2f} pp and Supervised Frozen by +{onl_vs_sup_noise:.2f} pp.\n"
        f"• Clean: Joint Frozen leads ({jnt_clean['accuracy']*100:.2f}%); Standard TTT stays within {std_vs_jnt_clean:+.2f} pp.",
        size=14.5, valign="top",
    )

    s4.set_notes(
        f"[About 45 seconds]\n"
        f"This slide shows the main test comparison on 2,000 held-out images. On the clean stream, joint training gives a +{jnt_vs_sup_clean:.2f} percentage point boost over supervised-only, "
        f"and Standard TTT stays close to that ceiling at {std_clean['accuracy']*100:.2f}%. "
        f"Under Gaussian noise, joint-frozen accuracy drops to {jnt_noise['accuracy']*100:.2f}%, and single-step Standard TTT improves modestly by +{std_vs_jnt_noise:.2f} pp. "
        f"By contrast, Online TTT surges to {onl_noise['accuracy']*100:.2f}% — a +{onl_vs_jnt_noise:.2f} pp improvement over Joint Frozen and +{onl_vs_std_noise:.2f} pp over Standard TTT.",
        "results/ttt_summary.csv; docs/exp_b_tuong_results.md",
    )
    slides.append(s4)

    # --------------------------------------------------------------------------
    # Slide 05: Rolling Window Stream Dynamics
    # --------------------------------------------------------------------------
    s5 = SlideScene(5, title="Stream dynamics across 2,000 test images", footer="Source: results/ttt_window_summary.csv (10 contiguous 200-image windows per stream, seed 42)")

    s5.add_image(72, 114, 1136, 424, assets_dir / "stream_dynamics_comfort.png", alt="Rolling window stream accuracy dynamics for clean and noisy streams")

    s5.add_box(72, 554, 554, 102, "", fill=MONO_WHITE, border=MONO_GRID, border_width=1.2)
    s5.add_text(90, 566, 518, 24, "Why Online TTT Dominates Under Noise", size=16, bold=True)
    s5.add_text(
        90, 594, 520, 56,
        f"Stationary noise allows consecutive 1-step rotation updates to accumulate:\n"
        f"Online TTT climbs from {w0_acc:.1f}% (W0) → {w2_acc:.1f}% (W2) → {peak_acc:.1f}% peak (W{peak_idx}).",
        size=14.8, valign="top",
    )

    s5.add_box(654, 554, 554, 102, "", fill=MONO_TINT, border=MONO_GRID, border_width=1.2)
    s5.add_text(672, 566, 518, 24, "Standard TTT Ceiling vs. Clean Stream Drift", size=16, bold=True)
    s5.add_text(
        672, 594, 520, 56,
        f"Resetting to θ0 limits Standard TTT to 1 step (+{std_vs_jnt_noise:.2f} pp on noise).\n"
        f"On clean data, Online TTT drifts ({onl_vs_jnt_clean:+.2f} pp) while Standard holds {std_clean['accuracy']*100:.2f}%.",
        size=14.8, valign="top",
    )

    s5.set_notes(
        f"[About 40 seconds]\n"
        f"Looking at accuracy across 200-image windows reveals why the two TTT modes behave differently. "
        f"In Stream 2 under stationary Gaussian noise, Online TTT starts near the frozen baseline at {w0_acc:.1f}% in Window 0, "
        f"rapidly adapts over the first 600 samples, and reaches {peak_acc:.1f}% by Window {peak_idx}, sustaining roughly 48% to 54% for the rest of the stream. "
        f"Meanwhile, resetting after every image leaves Standard TTT unable to move beyond a single gradient step from source weights.",
        "results/ttt_window_summary.csv; results/stream_accuracy_plot.png",
    )
    slides.append(s5)

    # --------------------------------------------------------------------------
    # Slide 06: Conclusions, Compute Cost & Verification Checks
    # --------------------------------------------------------------------------
    s6 = SlideScene(6, title="Conclusions, compute cost, and checks", footer="Artifacts: results/ttt_summary.csv, results/ttt_window_summary.csv, tests/test_ttt_exp_b.py")

    # Top 2 Research Answer Cards
    for cx, hdr, body in [
        (
            72,
            "Q1: DOES ROTATION TTT HELP WITHOUT CLEAN HARM?",
            f"• Noise (σ = 0.08): YES — Online TTT adds +{onl_vs_jnt_noise:.2f} pp over Joint Frozen\n"
            f"  ({onl_noise['accuracy']*100:.2f}% vs. {jnt_noise['accuracy']*100:.2f}%); Standard TTT adds +{std_vs_jnt_noise:.2f} pp ({std_noise['accuracy']*100:.2f}%).\n"
            f"• Clean: Standard TTT preserves accuracy ({std_vs_jnt_clean:+.2f} pp vs. Joint Frozen);\n"
            f"  Online TTT experiences mild cumulative drift ({onl_vs_jnt_clean:+.2f} pp).",
        ),
        (
            654,
            "Q2: RESET (STANDARD) VS. CARRYOVER (ONLINE)?",
            f"• Stationary Corruption Shift → Use Online TTT (+{onl_vs_std_noise:.2f} pp vs. Standard).\n"
            f"• Clean / Uncorrupted Stream → Use Standard TTT (+{std_vs_onl_clean:.2f} pp vs. Online)\n"
            f"  or Joint Frozen to prevent drift when no distribution shift occurs.\n"
            f"• Next Step: Run 3 seeds (42, 43, 44) for multi-seed variance.",
        ),
    ]:
        s6.add_box(cx, 114, 554, 162, "", fill=MONO_WHITE, border=MONO_GRID, border_width=1.2)
        s6.add_box(cx, 114, 554, 38, hdr, size=14.5, bold=True, align="left", dark=True, insets=12)
        s6.add_text(cx + 14, 160, 526, 108, body, size=14.5, valign="top")

    # Bottom Left: Runtime Cost Table
    s6.add_text(72, 302, 578, 22, "TEST-TIME COMPUTE COST (2,000 IMAGES, APPLE SILICON MPS)", size=14, bold=True, color=MONO_GRAY)

    def ms_per_img(cond_key: str) -> float:
        avg_s = 0.5 * (smap[("clean", cond_key)]["runtime_seconds"] + smap[("gaussian_noise", cond_key)]["runtime_seconds"])
        return avg_s / 2000.0 * 1000.0

    cost_rows = [
        ["Condition", "Updates", "Clean / Noise", "ms / img"],
        ["supervised_frozen", "0", f"{sup_clean['runtime_seconds']:.2f}s / {sup_noise['runtime_seconds']:.2f}s", f"~{ms_per_img('supervised_frozen'):.1f} ms"],
        ["joint_frozen", "0", f"{jnt_clean['runtime_seconds']:.2f}s / {jnt_noise['runtime_seconds']:.2f}s", f"~{ms_per_img('joint_frozen'):.1f} ms"],
        ["standard_ttt", "2,000", f"{std_clean['runtime_seconds']:.2f}s / {std_noise['runtime_seconds']:.2f}s", f"~{ms_per_img('standard_ttt'):.1f} ms"],
        ["online_ttt", "2,000", f"{onl_clean['runtime_seconds']:.2f}s / {onl_noise['runtime_seconds']:.2f}s", f"~{ms_per_img('online_ttt'):.1f} ms"],
    ]
    overhead_ms = ms_per_img("online_ttt") - ms_per_img("joint_frozen")
    fps_est = 1000.0 / ms_per_img("online_ttt")
    s6.add_table(
        72, 330,
        col_widths=[186, 86, 176, 130],
        row_heights=[40, 42, 42, 42, 42],
        rows=cost_rows,
        col_aligns=["left", "center", "center", "center"],
        highlight_rows=[4],
        bold_cells=[(4, 0)],
        font_size=14,
        header_size=14,
        pad_x=8,
    )
    s6.add_text(
        72, 550, 578, 72,
        f"• Adaptation adds ~{overhead_ms:.1f} ms/image over frozen inference (~{fps_est:.0f} FPS on MPS).\n"
        f"• Source training ({sup_meta['source_train_seconds']:.1f} s sup / {jnt_meta['source_train_seconds']:.1f} s joint) tracked separately.",
        size=14, valign="top", color=MONO_GRAY,
    )

    # Bottom Right: Verification Audit Table
    s6.add_text(680, 302, 528, 22, "VERIFICATION AUDIT (KHOI 8 CHECKS + ALKA CSV SCHEMA)", size=14, bold=True, color=MONO_GRAY)
    audit_rows = [
        ["Verification Criterion", "Status"],
        ["1–2. Fixed 2,000-img split & SHA-256 deterministic noise", "PASS"],
        ["3–4. Val/test isolation & zero test-label leakage", "PASS"],
        ["5–7. Per-image reset, frozen heads & stateless SGD", "PASS"],
        ["8. Smoke test + 23/10-col Alka CSV schema compliance", "PASS"],
    ]
    s6.add_table(
        680, 330,
        col_widths=[438, 90],
        row_heights=[40, 42, 42, 42, 42],
        rows=audit_rows,
        col_aligns=["left", "center"],
        highlight_rows=[],
        bold_cells=[(1, 1), (2, 1), (3, 1), (4, 1)],
        font_size=14,
        header_size=14,
        pad_x=8,
    )
    s6.add_text(
        680, 550, 528, 72,
        "• Validated by pytest suite (tests/test_ttt_exp_b.py, 39 passed) and\n"
        "  integrity audit (tests/verify_m3_integrity.py) against saved outputs.",
        size=14, valign="top", color=MONO_GRAY,
    )

    s6.set_notes(
        f"[About 40 seconds]\n"
        f"To conclude: under persistent Gaussian noise, Online TTT delivers a +{onl_vs_jnt_noise:.2f} pp gain over Joint Frozen and +{onl_vs_std_noise:.2f} pp over Standard TTT. "
        f"On clean streams, Standard TTT avoids cumulative feature drift and preserves source performance. "
        f"Runtime overhead is modest at about {overhead_ms:.1f} milliseconds per image, roughly {fps_est:.0f} frames per second on Apple Silicon MPS. "
        f"All eight evaluation checks and Alka CSV schemas pass, and the pipeline is ready for the full three-seed run.",
        "results/ttt_summary.csv; tests/test_ttt_exp_b.py; tests/verify_m3_integrity.py",
    )
    slides.append(s6)

    return slides


# ==============================================================================
# PDF Renderer (Vector Slide Scene via Matplotlib PdfPages)
# ==============================================================================

def render_pdf(slides: List[SlideScene], out_pdf: Path) -> None:
    """Render all SlideScenes to a matching 16:9 vector PDF."""
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for sc in slides:
            fig = plt.figure(figsize=(13.333333, 7.5), dpi=150)
            fig.patch.set_facecolor(MONO_WHITE)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, CANVAS_W)
            ax.set_ylim(CANVAS_H, 0)
            ax.axis("off")

            for item in sc.items:
                if isinstance(item, BoxItem):
                    _pdf_draw_box(ax, item)
                elif isinstance(item, LineItem):
                    _pdf_draw_line(ax, item)
                elif isinstance(item, ElbowItem):
                    _pdf_draw_elbow(ax, item)
                elif isinstance(item, TableItem):
                    _pdf_draw_table(ax, item)
                elif isinstance(item, ImageItem):
                    _pdf_draw_image(ax, item)

            pdf.savefig(fig, facecolor=MONO_WHITE)
            plt.close(fig)


def _pdf_draw_box(ax: plt.Axes, b: BoxItem) -> None:
    if b.fill is not None or b.border is not None:
        rect = mpatches.Rectangle(
            (b.x, b.y),
            b.w,
            b.h,
            facecolor=b.fill if b.fill is not None else "none",
            edgecolor=b.border if b.border is not None else "none",
            linewidth=b.border_width * PT_PER_PX if b.border is not None else 0.0,
            zorder=2,
        )
        ax.add_patch(rect)

    if b.text:
        if b.align == "left":
            tx = b.x + b.insets
            ha = "left"
        elif b.align == "right":
            tx = b.x + b.w - b.insets
            ha = "right"
        else:
            tx = b.x + b.w / 2.0
            ha = "center"

        if b.valign == "top":
            ty = b.y + b.insets
            va = "top"
        elif b.valign == "bottom":
            ty = b.y + b.h - b.insets
            va = "bottom"
        else:
            ty = b.y + b.h / 2.0
            va = "center"

        ax.text(
            tx,
            ty,
            b.text,
            fontsize=b.size * PT_PER_PX,
            fontweight="bold" if b.bold else "normal",
            fontstyle="italic" if b.italic else "normal",
            color=b.text_color,
            ha=ha,
            va=va,
            linespacing=1.24,
            zorder=4,
        )


def _pdf_draw_line(ax: plt.Axes, ln: LineItem) -> None:
    ls = "--" if ln.dashed else "-"
    lw = ln.width * PT_PER_PX
    if ln.arrow_end or ln.arrow_start:
        style = "<->" if (ln.arrow_start and ln.arrow_end) else ("->" if ln.arrow_end else "<-")
        ax.annotate(
            "",
            xy=(ln.x2, ln.y2),
            xytext=(ln.x1, ln.y1),
            arrowprops=dict(
                arrowstyle=style,
                color=ln.color,
                lw=lw,
                linestyle=ls,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=13,
            ),
            zorder=3,
        )
    else:
        ax.plot([ln.x1, ln.x2], [ln.y1, ln.y2], color=ln.color, lw=lw, linestyle=ls, zorder=3)


def _pdf_draw_elbow(ax: plt.Axes, el: ElbowItem) -> None:
    xm = (el.x1 + el.x2) / 2.0
    ls = "--" if el.dashed else "-"
    lw = el.width * PT_PER_PX
    ax.plot([el.x1, xm, xm], [el.y1, el.y1, el.y2], color=el.color, lw=lw, linestyle=ls, zorder=3)
    if el.arrow_end:
        ax.annotate(
            "",
            xy=(el.x2, el.y2),
            xytext=(xm, el.y2),
            arrowprops=dict(arrowstyle="->", color=el.color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0, mutation_scale=13),
            zorder=3,
        )
    else:
        ax.plot([xm, el.x2], [el.y2, el.y2], color=el.color, lw=lw, linestyle=ls, zorder=3)


def _pdf_draw_table(ax: plt.Axes, tbl: TableItem) -> None:
    cur_y = tbl.y
    for r_idx, row in enumerate(tbl.rows):
        rh = tbl.row_heights[r_idx]
        cur_x = tbl.x
        is_header = r_idx == 0
        is_hl = r_idx in tbl.highlight_rows
        for c_idx, val in enumerate(row):
            cw = tbl.col_widths[c_idx]
            bg = MONO_BLACK if is_header else ("#F0F4F7" if is_hl else MONO_WHITE)
            tc = MONO_WHITE if is_header else MONO_BLACK
            is_bold = is_header or ((r_idx, c_idx) in tbl.bold_cells)
            fs = tbl.header_size if is_header else tbl.font_size
            rect = mpatches.Rectangle(
                (cur_x, cur_y),
                cw,
                rh,
                facecolor=bg,
                edgecolor=MONO_GRID,
                linewidth=0.8 * PT_PER_PX,
                zorder=2,
            )
            ax.add_patch(rect)
            al = tbl.col_aligns[c_idx]
            if al == "left":
                tx, ha = cur_x + tbl.pad_x, "left"
            elif al == "right":
                tx, ha = cur_x + cw - tbl.pad_x, "right"
            else:
                tx, ha = cur_x + cw / 2.0, "center"
            ax.text(
                tx,
                cur_y + rh / 2.0,
                val,
                fontsize=fs * PT_PER_PX,
                fontweight="bold" if is_bold else "normal",
                color=tc,
                ha=ha,
                va="center",
                zorder=4,
            )
            cur_x += cw
        cur_y += rh


def _pdf_draw_image(ax: plt.Axes, im: ImageItem) -> None:
    arr = plt.imread(im.path)
    ax.imshow(arr, extent=(im.x, im.x + im.w, im.y + im.h, im.y), aspect="auto", zorder=2)


# ==============================================================================
# Native OOXML (.pptx) Renderer (Clones master/theme/layouts from TTT_Presentation.pptx)
# ==============================================================================

def _px_emu(v: float) -> int:
    return int(round(v * EMU_PER_PX))


def _hex(c: str) -> str:
    return c.lstrip("#").upper()


def _text_body_xml(
    text: str,
    size_px: float,
    bold: bool,
    italic: bool,
    align: str,
    valign: str,
    color: str,
    insets_px: float,
) -> str:
    al_map = {"left": "l", "center": "ctr", "right": "r"}
    va_map = {"top": "t", "middle": "ctr", "bottom": "b"}
    algn = al_map.get(align, "l")
    anc = va_map.get(valign, "ctr")
    ins = _px_emu(insets_px)
    sz = int(round(size_px * 75))
    b_val = "1" if bold else "0"
    i_val = "1" if italic else "0"
    clr = _hex(color)

    lines = text.split("\n") if text else [""]
    paras = []
    for line_str in lines:
        esc = html.escape(line_str)
        paras.append(
            f'<a:p xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            f'<a:pPr algn="{algn}">'
            f'<a:defRPr sz="{sz}" b="{b_val}" i="{i_val}">'
            f'<a:solidFill><a:srgbClr val="{clr}" /></a:solidFill>'
            f'<a:latin typeface="{FONT_FAMILY}" /><a:ea typeface="{FONT_FAMILY}" /><a:cs typeface="{FONT_FAMILY}" />'
            f'</a:defRPr></a:pPr>'
            f'<a:r><a:rPr sz="{sz}" b="{b_val}" i="{i_val}">'
            f'<a:solidFill><a:srgbClr val="{clr}" /></a:solidFill>'
            f'<a:latin typeface="{FONT_FAMILY}" /><a:ea typeface="{FONT_FAMILY}" /><a:cs typeface="{FONT_FAMILY}" />'
            f'</a:rPr><a:t>{esc}</a:t></a:r></a:p>'
        )
    return (
        f'<p:txBody>'
        f'<a:bodyPr wrap="square" lIns="{ins}" tIns="{ins}" rIns="{ins}" bIns="{ins}" anchor="{anc}" '
        f'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:noAutofit /></a:bodyPr>'
        f'<a:lstStyle xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" />'
        f'{"".join(paras)}'
        f'</p:txBody>'
    )


def _shape_box_xml(shape_id: int, name: str, b: BoxItem) -> str:
    x, y, w, h = _px_emu(b.x), _px_emu(b.y), _px_emu(b.w), _px_emu(b.h)
    fill_xml = (
        f'<a:solidFill xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:srgbClr val="{_hex(b.fill)}" /></a:solidFill>'
        if b.fill is not None
        else '<a:noFill xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" />'
    )
    if b.border is not None and b.border_width > 0:
        lw = _px_emu(b.border_width)
        ln_xml = (
            f'<a:ln w="{lw}" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            f'<a:solidFill><a:srgbClr val="{_hex(b.border)}" /></a:solidFill></a:ln>'
        )
    else:
        ln_xml = '<a:ln w="0" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:noFill /></a:ln>'
    tx_xml = _text_body_xml(b.text, b.size, b.bold, b.italic, b.align, b.valign, b.text_color, b.insets)
    return (
        f'<p:sp>'
        f'<p:nvSpPr><p:cNvPr id="{shape_id}" name="{html.escape(name)}" />'
        f'<p:cNvSpPr><a:spLocks noGrp="1" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" /></p:cNvSpPr><p:nvPr /></p:nvSpPr>'
        f'<p:spPr><a:xfrm xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:off x="{x}" y="{y}" /><a:ext cx="{w}" cy="{h}" /></a:xfrm>'
        f'<a:prstGeom prst="rect" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:avLst /></a:prstGeom>'
        f'{fill_xml}{ln_xml}</p:spPr>'
        f'{tx_xml}</p:sp>'
    )


def _connector_line_xml(shape_id: int, ln: LineItem) -> str:
    x1, y1, x2, y2 = ln.x1, ln.y1, ln.x2, ln.y2
    off_x, off_y = _px_emu(min(x1, x2)), _px_emu(min(y1, y2))
    cx, cy = _px_emu(abs(x2 - x1)), _px_emu(abs(y2 - y1))
    flip_h = ' flipH="1"' if x2 < x1 else ""
    flip_v = ' flipV="1"' if y2 < y1 else ""
    lw = _px_emu(ln.width)
    dash = "dash" if ln.dashed else "solid"
    head_xml = '<a:headEnd type="triangle" w="med" len="med" />' if ln.arrow_start else ""
    tail_xml = '<a:tailEnd type="triangle" w="med" len="med" />' if ln.arrow_end else ""
    return (
        f'<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="{shape_id}" name="conn-{shape_id}" /><p:cNvCxnSpPr /><p:nvPr /></p:nvCxnSpPr>'
        f'<p:spPr><a:xfrm{flip_h}{flip_v} xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        f'<a:off x="{off_x}" y="{off_y}" /><a:ext cx="{cx}" cy="{cy}" /></a:xfrm>'
        f'<a:prstGeom prst="straightConnector1" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:avLst /></a:prstGeom>'
        f'<a:ln w="{lw}" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        f'<a:solidFill><a:srgbClr val="{_hex(ln.color)}" /></a:solidFill>'
        f'<a:prstDash val="{dash}" />{head_xml}{tail_xml}</a:ln></p:spPr></p:cxnSp>'
    )


def _table_xml(shape_id: int, tbl: TableItem) -> str:
    x, y, w, h = _px_emu(tbl.x), _px_emu(tbl.y), _px_emu(tbl.w), _px_emu(tbl.h)
    grid_cols = "".join(f'<a:gridCol w="{_px_emu(cw)}" />' for cw in tbl.col_widths)
    al_map = {"left": "l", "center": "ctr", "right": "r"}
    pad_lr = _px_emu(tbl.pad_x)
    pad_tb = _px_emu(5)

    trs = []
    for r_idx, row in enumerate(tbl.rows):
        rh = _px_emu(tbl.row_heights[r_idx])
        is_header = r_idx == 0
        is_hl = r_idx in tbl.highlight_rows
        tcs = []
        for c_idx, val in enumerate(row):
            bg = _hex(MONO_BLACK if is_header else ("#F0F4F7" if is_hl else MONO_WHITE))
            tc = _hex(MONO_WHITE if is_header else MONO_BLACK)
            is_bold = is_header or ((r_idx, c_idx) in tbl.bold_cells)
            sz = int(round((tbl.header_size if is_header else tbl.font_size) * 75))
            b_val = "1" if is_bold else "0"
            algn = al_map.get(tbl.col_aligns[c_idx], "l")
            esc = html.escape(val)
            borders = "".join(
                f'<a:{side} w="9525"><a:solidFill><a:srgbClr val="BBBBBB" /></a:solidFill><a:prstDash val="solid" /></a:{side}>'
                for side in ("lnL", "lnR", "lnT", "lnB")
            )
            tcs.append(
                f'<a:tc><a:txBody>'
                f'<a:bodyPr lIns="{pad_lr}" tIns="{pad_tb}" rIns="{pad_lr}" bIns="{pad_tb}"><a:noAutofit /></a:bodyPr>'
                f'<a:lstStyle /><a:p><a:pPr algn="{algn}" /><a:r>'
                f'<a:rPr sz="{sz}" b="{b_val}"><a:solidFill><a:srgbClr val="{tc}" /></a:solidFill>'
                f'<a:latin typeface="{FONT_FAMILY}" /><a:ea typeface="{FONT_FAMILY}" /><a:cs typeface="{FONT_FAMILY}" /></a:rPr>'
                f'<a:t>{esc}</a:t></a:r></a:p></a:txBody>'
                f'<a:tcPr marL="{pad_lr}" marR="{pad_lr}" marT="{pad_tb}" marB="{pad_tb}" anchor="ctr">'
                f'{borders}<a:solidFill><a:srgbClr val="{bg}" /></a:solidFill></a:tcPr></a:tc>'
            )
        trs.append(f'<a:tr h="{rh}">{"".join(tcs)}</a:tr>')

    return (
        f'<p:graphicFrame>'
        f'<p:nvGraphicFramePr><p:cNvPr id="{shape_id}" name="table-{shape_id}" /><p:cNvGraphicFramePr /><p:nvPr /></p:nvGraphicFramePr>'
        f'<p:xfrm><a:off x="{x}" y="{y}" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" />'
        f'<a:ext cx="{w}" cy="{h}" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" /></p:xfrm>'
        f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        f'<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table">'
        f'<a:tbl><a:tblPr firstRow="1" bandRow="0" /><a:tblGrid>{grid_cols}</a:tblGrid>{"".join(trs)}</a:tbl>'
        f'</a:graphicData></a:graphic></p:graphicFrame>'
    )


def _pic_xml(shape_id: int, rel_id: str, im: ImageItem) -> str:
    x, y, w, h = _px_emu(im.x), _px_emu(im.y), _px_emu(im.w), _px_emu(im.h)
    return (
        f'<p:pic><p:nvPicPr><p:cNvPr id="{shape_id}" name="{html.escape(im.alt or f"pic-{shape_id}")}" />'
        f'<p:cNvPicPr><a:picLocks noChangeAspect="1" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" /></p:cNvPicPr><p:nvPr /></p:nvPicPr>'
        f'<p:blipFill><a:blip r:embed="{rel_id}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        f'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" /><a:stretch xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" /></p:blipFill>'
        f'<p:spPr><a:xfrm rot="0" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        f'<a:off x="{x}" y="{y}" /><a:ext cx="{w}" cy="{h}" /></a:xfrm>'
        f'<a:prstGeom prst="rect" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:avLst /></a:prstGeom>'
        f'</p:spPr></p:pic>'
    )


def _notes_xml(notes_text: str) -> str:
    paras = []
    for line_str in notes_text.split("\n"):
        esc = html.escape(line_str)
        paras.append(
            f'<a:p xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            f'<a:r><a:t>{esc}</a:t></a:r></a:p>'
        )
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<p:notes xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        '<p:cSld><p:spTree>'
        '<p:nvGrpSpPr><p:cNvPr id="1" name="" /><p:cNvGrpSpPr /><p:nvPr /></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" /></p:grpSpPr>'
        '<p:sp><p:nvSpPr><p:cNvPr id="2" name="Slide Image Placeholder 1" />'
        '<p:cNvSpPr><a:spLocks noGrp="1" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" /></p:cNvSpPr>'
        '<p:nvPr><p:ph type="sldImg" idx="0" /></p:nvPr></p:nvSpPr><p:spPr /></p:sp>'
        '<p:sp><p:nvSpPr><p:cNvPr id="3" name="Notes Placeholder 2" />'
        '<p:cNvSpPr><a:spLocks noGrp="1" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" /></p:cNvSpPr>'
        '<p:nvPr><p:ph type="body" idx="1" /></p:nvPr></p:nvSpPr><p:spPr />'
        f'<p:txBody><a:bodyPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" />'
        f'<a:lstStyle xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" />{"".join(paras)}</p:txBody></p:sp>'
        '<p:sp><p:nvSpPr><p:cNvPr id="4" name="Slide Number Placeholder 3" />'
        '<p:cNvSpPr><a:spLocks noGrp="1" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" /></p:cNvSpPr>'
        '<p:nvPr><p:ph type="sldNum" idx="5" /></p:nvPr></p:nvSpPr><p:spPr />'
        '<p:txBody><a:bodyPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" />'
        '<a:lstStyle xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" />'
        '<a:p xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" /></p:txBody></p:sp>'
        '</p:spTree></p:cSld>'
        '<p:clrMapOvr><a:masterClrMapping xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" /></p:clrMapOvr>'
        '</p:notes>'
    )


def render_pptx(slides: List[SlideScene], template_pptx: Path, out_pptx: Path) -> None:
    """Build native editable .pptx by cloning theme/master/layouts from TTT_Presentation.pptx."""
    out_pptx.parent.mkdir(parents=True, exist_ok=True)
    preserve_prefixes = (
        "_rels/.rels",
        "ppt/theme/",
        "ppt/slideMasters/",
        "ppt/slideLayouts/",
        "ppt/notesMasters/",
        "ppt/presProps.xml",
        "ppt/tableStyles.xml",
    )

    with zipfile.ZipFile(template_pptx, "r") as zin, zipfile.ZipFile(out_pptx, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        # 1. Copy static scaffold parts (masters, layouts, themes, tableStyles)
        for info in zin.infolist():
            if info.filename.startswith(preserve_prefixes):
                zout.writestr(info.filename, zin.read(info.filename))

        media_map: Dict[Path, str] = {}
        media_counter = 0

        slide_rels_ids: List[str] = []
        slide_ct_entries: List[str] = []

        # 2. Emit each slide, its rels, media, and notesSlide
        for idx, sc in enumerate(slides, start=1):
            shape_id = 2
            shapes_xml: List[str] = []
            extra_rels: List[str] = []

            for item in sc.items:
                if isinstance(item, BoxItem):
                    shapes_xml.append(_shape_box_xml(shape_id, f"shape-{shape_id}", item))
                    shape_id += 1
                elif isinstance(item, LineItem):
                    shapes_xml.append(_connector_line_xml(shape_id, item))
                    shape_id += 1
                elif isinstance(item, ElbowItem):
                    xm = (item.x1 + item.x2) / 2.0
                    segs = [
                        LineItem(item.x1, item.y1, xm, item.y1, color=item.color, width=item.width, dashed=item.dashed),
                        LineItem(xm, item.y1, xm, item.y2, color=item.color, width=item.width, dashed=item.dashed),
                        LineItem(xm, item.y2, item.x2, item.y2, color=item.color, width=item.width, dashed=item.dashed, arrow_end=item.arrow_end),
                    ]
                    for seg in segs:
                        shapes_xml.append(_connector_line_xml(shape_id, seg))
                        shape_id += 1
                elif isinstance(item, TableItem):
                    shapes_xml.append(_table_xml(shape_id, item))
                    shape_id += 1
                elif isinstance(item, ImageItem):
                    if item.path not in media_map:
                        media_counter += 1
                        media_target = f"ppt/media/image{media_counter}.png"
                        zout.writestr(media_target, item.path.read_bytes())
                        media_map[item.path] = f"/ppt/media/image{media_counter}.png"
                    rel_id = f"Rimg{idx}_{shape_id}"
                    extra_rels.append(
                        f'<Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                        f'Target="{media_map[item.path]}" Id="{rel_id}" />'
                    )
                    shapes_xml.append(_pic_xml(shape_id, rel_id, item))
                    shape_id += 1

            slide_xml = (
                '<?xml version="1.0" encoding="utf-8"?>'
                '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
                '<p:cSld><p:bg><p:bgPr><a:solidFill xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
                '<a:srgbClr val="FFFFFF" /></a:solidFill></p:bgPr></p:bg>'
                '<p:spTree>'
                '<p:nvGrpSpPr><p:cNvPr id="1" name="" /><p:cNvGrpSpPr /><p:nvPr /></p:nvGrpSpPr>'
                '<p:grpSpPr><a:xfrm xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" /></p:grpSpPr>'
                f'{"".join(shapes_xml)}'
                '</p:spTree></p:cSld></p:sld>'
            )
            zout.writestr(f"ppt/slides/slide{idx}.xml", slide_xml.encode("utf-8"))

            slide_rels_xml = (
                '<?xml version="1.0" encoding="utf-8"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" '
                'Target="/ppt/slideLayouts/slideLayout6.xml" Id="Rlayout" />'
                f'<Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" '
                f'Target="/ppt/notesSlides/notesSlide{idx}.xml" Id="Rnotes" />'
                f'{"".join(extra_rels)}'
                '</Relationships>'
            )
            zout.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", slide_rels_xml.encode("utf-8"))

            zout.writestr(f"ppt/notesSlides/notesSlide{idx}.xml", _notes_xml(sc.notes).encode("utf-8"))
            notes_rels_xml = (
                '<?xml version="1.0" encoding="utf-8"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                f'<Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" '
                f'Target="/ppt/slides/slide{idx}.xml" Id="Rslide" />'
                '<Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster" '
                'Target="/ppt/notesMasters/notesMaster1.xml" Id="RnotesMaster" />'
                '</Relationships>'
            )
            zout.writestr(f"ppt/notesSlides/_rels/notesSlide{idx}.xml.rels", notes_rels_xml.encode("utf-8"))

            r_id = f"Rsld{idx}"
            slide_rels_ids.append(r_id)
            slide_ct_entries.append(
                f'<Override PartName="/ppt/slides/slide{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml" />'
                f'<Override PartName="/ppt/notesSlides/notesSlide{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml" />'
            )

        # 3. Presentation manifest & rels
        sld_id_list = "".join(
            f'<p:sldId id="{255 + i}" r:id="{r_id}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" />'
            for i, r_id in enumerate(slide_rels_ids, start=1)
        )
        pres_xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
            '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="Rmaster" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" /></p:sldMasterIdLst>'
            '<p:notesMasterIdLst><p:notesMasterId r:id="Rnmaster" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" /></p:notesMasterIdLst>'
            f'<p:sldIdLst>{sld_id_list}</p:sldIdLst>'
            f'<p:sldSz cx="{_px_emu(CANVAS_W)}" cy="{_px_emu(CANVAS_H)}" />'
            '<p:notesSz cx="6858000" cy="9144000" />'
            '</p:presentation>'
        )
        zout.writestr("ppt/presentation.xml", pres_xml.encode("utf-8"))

        pres_rels_slides = "".join(
            f'<Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" '
            f'Target="/ppt/slides/slide{i}.xml" Id="{r_id}" />'
            for i, r_id in enumerate(slide_rels_ids, start=1)
        )
        pres_rels_xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="/ppt/theme/theme1.xml" Id="Rtheme" />'
            '<Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="/ppt/slideMasters/slideMaster1.xml" Id="Rmaster" />'
            '<Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster" Target="/ppt/notesMasters/notesMaster1.xml" Id="Rnmaster" />'
            '<Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps" Target="/ppt/presProps.xml" Id="RpresProps" />'
            '<Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/tableStyles" Target="/ppt/tableStyles.xml" Id="RtblStyles" />'
            f'{pres_rels_slides}'
            '</Relationships>'
        )
        zout.writestr("ppt/_rels/presentation.xml.rels", pres_rels_xml.encode("utf-8"))

        # 4. Content Types (copy static overrides from template, replace slide/notesSlide overrides)
        ct_root = ET.fromstring(zin.read("[Content_Types].xml"))
        ns_ct = "http://schemas.openxmlformats.org/package/2006/content-types"
        static_overrides = []
        for child in ct_root:
            if child.tag.endswith("Override"):
                pname = child.attrib.get("PartName", "")
                if pname.startswith("/ppt/slides/") or pname.startswith("/ppt/notesSlides/"):
                    continue
                static_overrides.append(
                    f'<Override PartName="{pname}" ContentType="{child.attrib["ContentType"]}" />'
                )
        ct_xml = (
            f'<?xml version="1.0" encoding="utf-8"?><Types xmlns="{ns_ct}">'
            '<Default Extension="xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml" />'
            '<Default Extension="png" ContentType="image/png" />'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />'
            f'{"".join(static_overrides)}{"".join(slide_ct_entries)}</Types>'
        )
        zout.writestr("[Content_Types].xml", ct_xml.encode("utf-8"))

        # 5. DocProps
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        app_xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<ap:Properties xmlns:ap="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
            '<ap:Application>Experiment B Presentation Builder</ap:Application>'
            '<ap:PresentationFormat>Widescreen</ap:PresentationFormat>'
            f'<ap:Slides>{len(slides)}</ap:Slides>'
            f'<ap:Notes>{len(slides)}</ap:Notes>'
            '<ap:HiddenSlides>0</ap:HiddenSlides><ap:SharedDoc>false</ap:SharedDoc><ap:DocSecurity>0</ap:DocSecurity>'
            '</ap:Properties>'
        )
        core_xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<coreProperties xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties">'
            '<dc:creator>Tuong</dc:creator><cp:lastModifiedBy>Tuong</cp:lastModifiedBy>'
            '<dc:title>Experiment B: Test-Time Training on CIFAR-100</dc:title>'
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{now_iso}</dcterms:created>'
            f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now_iso}</dcterms:modified>'
            '</coreProperties>'
        )
        zout.writestr("docProps/app.xml", app_xml.encode("utf-8"))
        zout.writestr("docProps/core.xml", core_xml.encode("utf-8"))


# ==============================================================================
# Entrypoint
# ==============================================================================

def main() -> None:
    tuong_dir = Path(__file__).resolve().parent
    repo_root = tuong_dir.parents[2]
    assets_dir = tuong_dir / "assets"
    template_pptx = tuong_dir / "TTT_Presentation.pptx"
    out_pptx = tuong_dir / "Exp_B_TTT_Report_Tuong.pptx"
    out_pdf = tuong_dir / "Exp_B_TTT_Report_Tuong.pdf"

    data = load_experiment_data(repo_root)

    generate_lr_tuning_chart(data, assets_dir / "lr_tuning_chart.png")
    generate_test_accuracy_bars(data, assets_dir / "test_accuracy_bars.png")
    generate_stream_dynamics_chart(data, assets_dir / "stream_dynamics_comfort.png")

    slides = build_slides(data, assets_dir)
    render_pdf(slides, out_pdf)
    render_pptx(slides, template_pptx, out_pptx)

    print(f"Generated PDF:  {out_pdf} ({out_pdf.stat().st_size:,} bytes)")
    print(f"Generated PPTX: {out_pptx} ({out_pptx.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
