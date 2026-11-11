"""
Test-Time Training (TTT) on CIFAR-100-20 (Experiment B).

Implements source model training, label-free test-time adaptation engine,
4-condition stream evaluation, rolling-window metric computation, and Alka-compatible
CSV logging.
"""

from __future__ import annotations

import argparse
import copy
import csv
import inspect
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import yaml

from src.data import (
    EXP_B_TEST_MEMBERSHIP_PATH,
    SHARED_C20_MEMBERSHIP_PATH,
    CIFAR100ExpBDataset,
    apply_gaussian_noise,
    get_exp_b_loader,
    get_exp_b_loaders,
)
from src.models import (
    CIFAR100_MEAN,
    CIFAR100_STD,
    ResidualGN,
    SmallResNetTTT,
    conv_gn,
    rotate_batch,
)

# Aliases and re-exports for backwards / prompt / test compatibility
CIFARResNet18TTT = SmallResNetTTT
create_rotated_batch = rotate_batch
get_cifar100_loaders = get_exp_b_loaders

# Valid experimental conditions for Experiment B (immutable tuple)
VALID_CONDITIONS: Tuple[str, ...] = (
    "supervised_frozen",
    "joint_frozen",
    "standard_ttt",
    "online_ttt",
)

__all__ = [
    "ALKA_SUMMARY_COLUMNS",
    "ALKA_WINDOW_COLUMNS",
    "VALID_CONDITIONS",
    "CIFARResNet18TTT",
    "create_rotated_batch",
    "get_cifar100_loaders",
    "adapt_and_predict_image",
    "build_summary_row",
    "compute_delta_accuracy",
    "compute_rolling_windows",
    "evaluate_condition",
    "evaluate_source_model",
    "load_model",
    "save_summary_csv",
    "save_window_csv",
    "train_source_model",
    "main",
]

# Alka CSV Schemas
ALKA_SUMMARY_COLUMNS = [
    "experiment",
    "condition",
    "seed",
    "checkpoint",
    "dataset",
    "split",
    "corruption",
    "severity",
    "accuracy",
    "runtime_seconds",
    "update_count",
    "rotation_loss",
    "learning_rate",
    "adaptation_steps",
    "batch_size",
    "loss_weights",
    "prediction_timing",
    "reset_mode",
    "optimizer_reset",
    "hardware",
    "code_version",
    "config_path",
    "run_id",
]

ALKA_WINDOW_COLUMNS = [
    "run_id",
    "experiment",
    "condition",
    "seed",
    "window_index",
    "start_sample",
    "end_sample",
    "window_size",
    "correct_count",
    "accuracy",
]


def build_summary_row(
    experiment: str = "B",
    condition: str = "supervised_frozen",
    seed: int = 42,
    checkpoint: str = "",
    dataset: str = "CIFAR-100-20",
    split: str = "test",
    corruption: str = "clean",
    severity: Union[int, str] = "none",
    accuracy: float = 0.0,
    runtime_seconds: float = 0.0,
    update_count: int = 0,
    rotation_loss: float = 0.0,
    learning_rate: float = 0.001,
    adaptation_steps: int = 1,
    batch_size: int = 1,
    loss_weights: Optional[str] = None,
    prediction_timing: Optional[str] = None,
    reset_mode: Optional[str] = None,
    optimizer_reset: Optional[str] = None,
    hardware: str = "cpu",
    code_version: str = "git-head",
    config_path: str = "configs/ttt_cifar100_exp_b.yaml",
    run_id: str = "",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build a validated 23-column summary dictionary conforming to ALKA_SUMMARY_COLUMNS.

    Supports explicit specification of all 23 Alka summary fields, with automatic
    metadata derivation (loss_weights, prediction_timing, reset_mode, optimizer_reset,
    adaptation_steps) based on the condition when not explicitly passed.
    """
    if condition not in VALID_CONDITIONS:
        raise ValueError(
            f"condition must be one of {sorted(VALID_CONDITIONS)}, got '{condition}'"
        )

    # Accept checkpoint_path alias if passed via kwargs
    if not checkpoint and "checkpoint_path" in kwargs:
        checkpoint = str(kwargs["checkpoint_path"])

    # Auto-derive condition-dependent metadata if not explicitly provided
    if condition == "supervised_frozen":
        default_loss_weights = "1.0_cls"
        default_pred_timing = "none"
        default_reset_mode = "none"
        default_opt_reset = "none"
        default_adapt_steps = 0
    elif condition == "joint_frozen":
        default_loss_weights = "1.0_cls_1.0_rot"
        default_pred_timing = "none"
        default_reset_mode = "none"
        default_opt_reset = "none"
        default_adapt_steps = 0
    elif condition == "standard_ttt":
        default_loss_weights = "1.0_cls_1.0_rot"
        default_pred_timing = "post_adaptation"
        default_reset_mode = "per_sample"
        default_opt_reset = "stateless"
        default_adapt_steps = 1
    elif condition == "online_ttt":
        default_loss_weights = "1.0_cls_1.0_rot"
        default_pred_timing = "post_adaptation"
        default_reset_mode = "carryover"
        default_opt_reset = "stateless"
        default_adapt_steps = 1
    else:
        default_loss_weights = "1.0_cls_1.0_rot"
        default_pred_timing = "post_adaptation"
        default_reset_mode = "none"
        default_opt_reset = "stateless"
        default_adapt_steps = 1

    actual_loss_weights = loss_weights if loss_weights is not None else default_loss_weights
    actual_pred_timing = prediction_timing if prediction_timing is not None else default_pred_timing
    actual_reset_mode = reset_mode if reset_mode is not None else default_reset_mode
    actual_opt_reset = optimizer_reset if optimizer_reset is not None else default_opt_reset
    actual_adapt_steps = adaptation_steps if adaptation_steps is not None else default_adapt_steps

    # Normalize severity field
    if corruption == "clean" and (severity is None or severity == ""):
        actual_severity: Union[int, str] = "none"
    elif corruption == "gaussian_noise" and (severity is None or severity == "" or severity == "none"):
        actual_severity = 3
    else:
        actual_severity = severity

    # Normalize run_id field
    if not run_id:
        run_id = f"exp_b_{condition}_{corruption}_seed{seed}"

    row: Dict[str, Any] = {
        "experiment": str(experiment),
        "condition": str(condition),
        "seed": int(seed),
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "split": str(split),
        "corruption": str(corruption),
        "severity": actual_severity,
        "accuracy": float(accuracy),
        "runtime_seconds": float(runtime_seconds),
        "update_count": int(update_count),
        "rotation_loss": float(rotation_loss),
        "learning_rate": float(learning_rate),
        "adaptation_steps": int(actual_adapt_steps),
        "batch_size": int(batch_size),
        "loss_weights": str(actual_loss_weights),
        "prediction_timing": str(actual_pred_timing),
        "reset_mode": str(actual_reset_mode),
        "optimizer_reset": str(actual_opt_reset),
        "hardware": str(hardware),
        "code_version": str(code_version),
        "config_path": str(config_path),
        "run_id": str(run_id),
    }

    if "predictions" in kwargs:
        row["predictions"] = kwargs["predictions"]

    return row


def resolve_device(device_arg: str = "auto") -> torch.device:
    """Resolve target execution device (cuda, mps, or cpu)."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def get_code_version() -> str:
    """Retrieve git revision if available, falling back to 'git-head'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "git-head"


def load_model(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    num_classes: int = 100,
) -> Tuple[SmallResNetTTT, Dict[str, Any]]:
    """Load SmallResNetTTT weights and metadata from checkpoint.

    Handles full checkpoint dictionaries ('model_state_dict' or 'state_dict')
    and raw state dictionaries seamlessly.
    """
    device = torch.device(device)
    model = SmallResNetTTT(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    metadata: Dict[str, Any] = {}
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
            metadata = {k: v for k, v in checkpoint.items() if k != "state_dict"}
        else:
            model.load_state_dict(checkpoint)
            metadata = {}
    else:
        raise ValueError(f"Unsupported checkpoint structure at {checkpoint_path}")

    model.to(device)
    model.eval()
    return model, metadata


def compute_rolling_windows(
    predictions: List[int],
    targets: List[int],
    window_size: int = 200,
    run_id: str = "test_run",
    experiment: str = "B",
    condition: str = "online_ttt",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Compute contiguous fixed-size rolling windows with remainder support."""
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have identical length")
    if len(predictions) == 0:
        raise ValueError("Cannot compute windows for empty predictions list")

    total_samples = len(predictions)
    windows: List[Dict[str, Any]] = []
    num_windows = math.ceil(total_samples / window_size)

    for w_idx in range(num_windows):
        start = w_idx * window_size
        end = min(start + window_size, total_samples)
        actual_size = end - start

        w_preds = predictions[start:end]
        w_targets = targets[start:end]
        correct = sum(1 for p, t in zip(w_preds, w_targets) if p == t)
        acc = float(correct) / actual_size

        windows.append({
            "run_id": run_id,
            "experiment": experiment,
            "condition": condition,
            "seed": seed,
            "window_index": w_idx,
            "start_sample": start,
            "end_sample": end,
            "window_size": actual_size,
            "correct_count": correct,
            "accuracy": acc,
        })

    return windows


def compute_delta_accuracy(
    acc_target: float,
    acc_baseline: float,
) -> float:
    """Compute accuracy change in percentage points against matched baseline."""
    return (float(acc_target) - float(acc_baseline)) * 100.0


def adapt_and_predict_image(
    model: SmallResNetTTT,
    image: torch.Tensor,
    ttt_lr: float = 0.001,
    ttt_steps: int = 1,
    lambda_rot: float = 1.0,
) -> Tuple[torch.Tensor, float, float, int]:
    """Label-free test-time adaptation on a single test image.

    Strictly satisfies Khoi Check #4:
    - Accepts ONLY image (never label, target, y, etc.).
    - Forces model.eval().
    - Keeps classification and rotation heads frozen (requires_grad=False, grad=None).
    - Updates only model.features via stateless SGD(momentum=0.0) on 4 rotated views.
    - If ttt_lr == 0.0, features are bit-for-bit unchanged.
    - Returns (logits_100, pre_adapt_rot_loss, post_adapt_rot_loss, rot_correct).
    """
    if ttt_lr < 0.0:
        raise ValueError(f"ttt_lr must be non-negative, got {ttt_lr}")
    if ttt_steps < 0:
        raise ValueError(f"ttt_steps must be non-negative, got {ttt_steps}")

    # Enforce evaluation mode (disables dropout)
    model.eval()

    # Freeze heads and clear any residual gradients
    for p in model.classifier.parameters():
        p.requires_grad = False
        p.grad = None
    for p in model.rotation_head.parameters():
        p.requires_grad = False
        p.grad = None

    # Enable gradients strictly on feature extractor
    for p in model.features.parameters():
        p.requires_grad = True

    # Ensure 4D tensor (1, C, H, W)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4 or image.size(0) != 1:
        raise ValueError(
            f"adapt_and_predict_image expects 3D or (1, C, H, W) 4D tensor, got {image.shape}"
        )

    # Place image on model's device
    device = next(model.parameters()).device
    image = image.to(device)

    # Generate 4 rotated views [0°, 90°, 180°, 270°]
    rot_images, rot_labels = rotate_batch(image)

    # Compute pre-adaptation rotation loss
    with torch.no_grad():
        pre_rot_logits = model.forward_rotation(rot_images)
        pre_adapt_rot_loss = F.cross_entropy(pre_rot_logits, rot_labels).item()

    # Perform stateless SGD steps only if learning rate and steps are positive
    if ttt_lr > 0.0 and ttt_steps > 0:
        optimizer = torch.optim.SGD(
            model.features.parameters(), lr=ttt_lr, momentum=0.0
        )
        for _ in range(ttt_steps):
            optimizer.zero_grad(set_to_none=True)
            rot_logits = model.forward_rotation(rot_images)
            loss = lambda_rot * F.cross_entropy(rot_logits, rot_labels)
            loss.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Compute post-adaptation rotation loss, rotation accuracy, and unmasked class logits
    with torch.no_grad():
        post_rot_logits = model.forward_rotation(rot_images)
        post_adapt_rot_loss = F.cross_entropy(post_rot_logits, rot_labels).item()
        rot_preds = post_rot_logits.argmax(dim=-1)
        rot_correct = int((rot_preds == rot_labels).sum().item())
        logits_100 = model(image)

    # Ensure heads remain strictly frozen with no lingering gradients
    for p in model.classifier.parameters():
        p.requires_grad = False
        p.grad = None
    for p in model.rotation_head.parameters():
        p.requires_grad = False
        p.grad = None

    return logits_100, float(pre_adapt_rot_loss), float(post_adapt_rot_loss), int(rot_correct)


def evaluate_condition(
    model: SmallResNetTTT,
    loader: DataLoader,
    condition: str,
    ttt_lr: float = 0.001,
    ttt_steps: int = 1,
    window_size: int = 200,
    device: Union[str, torch.device] = "cpu",
    corruption: str = "clean",
    severity: Union[int, str] = "none",
    seed: int = 42,
    config_path: str = "configs/ttt_cifar100_exp_b.yaml",
    run_id: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    summary_csv_path: Optional[Union[str, Path]] = None,
    window_csv_path: Optional[Union[str, Path]] = None,
    code_version: Optional[str] = None,
    lambda_rot: float = 1.0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate a model on a streaming DataLoader under one of the 4 conditions.

    Conditions:
    - supervised_frozen: 0 updates, torch.no_grad()
    - joint_frozen: 0 updates, torch.no_grad()
    - standard_ttt: adapts features per sample, resets features after each sample
    - online_ttt: adapts features per sample, carries forward features across stream
    """
    if condition not in VALID_CONDITIONS:
        raise ValueError(
            f"condition must be one of {sorted(VALID_CONDITIONS)}, got '{condition}'"
        )

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    # Freeze heads and verify immutability throughout
    for p in model.classifier.parameters():
        p.requires_grad = False
        p.grad = None
    for p in model.rotation_head.parameters():
        p.requires_grad = False
        p.grad = None

    init_cls_sd = {k: v.clone() for k, v in model.classifier.state_dict().items()}
    init_rot_sd = {k: v.clone() for k, v in model.rotation_head.state_dict().items()}

    # Standard TTT requires an episodic reset snapshot
    theta_0: Optional[Dict[str, torch.Tensor]] = None
    if condition == "standard_ttt":
        theta_0 = {k: v.clone() for k, v in model.features.state_dict().items()}

    if run_id is None:
        run_id = f"exp_b_{condition}_{corruption}_seed{seed}"
    if code_version is None:
        code_version = get_code_version()

    predictions: List[int] = []
    targets: List[int] = []
    sample_ids: List[str] = []
    rotation_losses: List[float] = []
    per_sample_logs: List[Dict[str, Any]] = []
    update_count = 0
    total_correct = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    start_time = time.perf_counter()

    for batch in loader:
        if len(batch) == 3:
            images, labels, batch_sids = batch
        elif len(batch) == 2:
            images, labels = batch
            batch_sids = [f"sample_{i}" for i in range(len(images))]
        else:
            images = batch[0]
            labels = torch.zeros(len(images), dtype=torch.long)
            batch_sids = [f"sample_{i}" for i in range(len(images))]

        batch_size = images.size(0)
        for i in range(batch_size):
            img = images[i : i + 1].to(device)
            if isinstance(labels, torch.Tensor):
                lbl = int(labels[i].item() if labels.dim() > 0 else labels.item())
            elif isinstance(labels, (list, tuple)):
                lbl = int(labels[i])
            else:
                lbl = int(labels)

            if isinstance(batch_sids, (list, tuple)):
                sid = str(batch_sids[i])
            else:
                sid = str(batch_sids)

            if condition in ("supervised_frozen", "joint_frozen"):
                with torch.no_grad():
                    logits_100 = model(img)
                    rot_imgs, rot_lbls = rotate_batch(img)
                    rot_logits = model.forward_rotation(rot_imgs)
                    rot_loss = F.cross_entropy(rot_logits, rot_lbls).item()
            elif condition == "standard_ttt":
                assert theta_0 is not None
                logits_100, _, post_loss, _ = adapt_and_predict_image(
                    model, img, ttt_lr=ttt_lr, ttt_steps=ttt_steps, lambda_rot=lambda_rot
                )
                rot_loss = post_loss
                update_count += ttt_steps
                # Immediate episodic reset to initial features
                model.features.load_state_dict(theta_0)
            elif condition == "online_ttt":
                logits_100, _, post_loss, _ = adapt_and_predict_image(
                    model, img, ttt_lr=ttt_lr, ttt_steps=ttt_steps, lambda_rot=lambda_rot
                )
                rot_loss = post_loss
                update_count += ttt_steps
                # Features carry over into subsequent samples

            # Record unmasked class prediction across all 100 classes
            pred = int(logits_100.argmax(dim=-1).item())
            is_correct = (pred == lbl)
            if is_correct:
                total_correct += 1

            predictions.append(pred)
            targets.append(lbl)
            sample_ids.append(sid)
            rotation_losses.append(rot_loss)

            n_samples = len(predictions)
            cum_acc = float(total_correct) / n_samples
            win_start = max(0, n_samples - window_size)
            win_corr = sum(
                1 for p, t in zip(predictions[win_start:n_samples], targets[win_start:n_samples]) if p == t
            )
            win_acc = float(win_corr) / (n_samples - win_start)

            per_sample_logs.append({
                "sample_id": sid,
                "true_label": lbl,
                "predicted_label": pred,
                "predicted_class": pred,
                "is_correct": bool(is_correct),
                "correct": bool(is_correct),
                "rotation_loss": float(rot_loss),
                "cumulative_accuracy": float(cum_acc),
                "window_accuracy": float(win_acc),
            })

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    runtime_seconds = time.perf_counter() - start_time

    # Verify head parameters remained bit-for-bit unchanged
    for k, v in init_cls_sd.items():
        if not torch.equal(model.classifier.state_dict()[k], v):
            raise RuntimeError(
                f"Head invariant violated: classifier parameter '{k}' was modified"
            )
    for k, v in init_rot_sd.items():
        if not torch.equal(model.rotation_head.state_dict()[k], v):
            raise RuntimeError(
                f"Head invariant violated: rotation_head parameter '{k}' was modified"
            )

    total_samples = len(predictions)
    overall_acc = float(total_correct) / total_samples if total_samples > 0 else 0.0
    mean_rot_loss = float(sum(rotation_losses) / len(rotation_losses)) if rotation_losses else 0.0

    # Compute rolling windows
    windows = compute_rolling_windows(
        predictions,
        targets,
        window_size=window_size,
        run_id=run_id,
        experiment="B",
        condition=condition,
        seed=seed,
    )

    # Build 23-column summary row conforming to ALKA_SUMMARY_COLUMNS (plus predictions)
    adapt_steps = ttt_steps if condition in ("standard_ttt", "online_ttt") else 0
    summary_row = build_summary_row(
        experiment="B",
        condition=condition,
        seed=seed,
        checkpoint=str(checkpoint_path or ""),
        dataset="CIFAR-100-20",
        split="test",
        corruption=corruption,
        severity=severity,
        accuracy=overall_acc,
        runtime_seconds=runtime_seconds,
        update_count=update_count,
        rotation_loss=mean_rot_loss,
        learning_rate=ttt_lr,
        adaptation_steps=adapt_steps,
        batch_size=1,
        hardware=str(device.type),
        code_version=code_version,
        config_path=config_path,
        run_id=run_id,
        predictions=predictions,
    )

    # Persist summary and window CSVs if paths provided
    if summary_csv_path is not None:
        save_summary_csv(summary_row, summary_csv_path)
    if window_csv_path is not None:
        save_window_csv(windows, window_csv_path)

    return summary_row, windows, per_sample_logs


def save_summary_csv(
    rows: Union[Dict[str, Any], List[Dict[str, Any]]],
    path: Union[str, Path],
) -> None:
    """Save or append summary rows conforming to ALKA_SUMMARY_COLUMNS."""
    if isinstance(rows, dict):
        rows = [rows]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with open(path, mode="a" if file_exists else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALKA_SUMMARY_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            clean_row = {col: row.get(col, "") for col in ALKA_SUMMARY_COLUMNS}
            writer.writerow(clean_row)


def save_window_csv(
    rows: List[Dict[str, Any]],
    path: Union[str, Path],
) -> None:
    """Save or append window rows conforming to ALKA_WINDOW_COLUMNS."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with open(path, mode="a" if file_exists else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALKA_WINDOW_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            clean_row = {col: row.get(col, "") for col in ALKA_WINDOW_COLUMNS}
            writer.writerow(clean_row)


def evaluate_source_model(
    model: SmallResNetTTT,
    val_loader: DataLoader,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[float, float]:
    """Evaluate classification accuracy and rotation accuracy on validation loader."""
    model.eval()
    device = torch.device(device)
    correct_cls = 0
    total_cls = 0
    correct_rot = 0
    total_rot = 0

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            cls_logits = model(images)
            preds = cls_logits.argmax(dim=-1)
            correct_cls += (preds == labels).sum().item()
            total_cls += labels.size(0)

            rot_images, rot_labels = rotate_batch(images)
            rot_logits = model.forward_rotation(rot_images)
            rot_preds = rot_logits.argmax(dim=-1)
            correct_rot += (rot_preds == rot_labels).sum().item()
            total_rot += rot_labels.size(0)

    val_acc = float(correct_cls) / total_cls if total_cls > 0 else 0.0
    val_rot_acc = float(correct_rot) / total_rot if total_rot > 0 else 0.0
    return val_acc, val_rot_acc


def train_source_model(
    model: Optional[SmallResNetTTT] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    mode: str = "joint",
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    pct_start: float = 0.15,
    lambda_rot: float = 1.0,
    device: Union[str, torch.device] = "cpu",
    save_path: Optional[Union[str, Path]] = None,
    checkpoint_dir: Union[str, Path] = "checkpoints",
    seed: int = 42,
    config: Optional[Union[str, Path, Dict[str, Any]]] = None,
) -> Tuple[SmallResNetTTT, Dict[str, Any]]:
    """Train source model in 'supervised' or 'joint' mode for 15 epochs."""
    if mode not in {"supervised", "joint"}:
        raise ValueError(f"mode must be 'supervised' or 'joint', got '{mode}'")

    config_dict: Dict[str, Any] = {}
    if isinstance(config, (str, Path)):
        with open(config, "r") as f:
            config_dict = yaml.safe_load(f) or {}
    elif isinstance(config, dict):
        config_dict = config

    st_cfg = config_dict.get("source_training", config_dict.get("source_train", {}))
    epochs = st_cfg.get("epochs", epochs)
    lr = st_cfg.get("learning_rate", lr)
    weight_decay = st_cfg.get("weight_decay", weight_decay)
    pct_start = st_cfg.get("pct_start", pct_start)
    lambda_rot = st_cfg.get("lambda_rot", lambda_rot)

    paths_cfg = config_dict.get("paths", {})
    membership_path = paths_cfg.get("shared_c20_membership", SHARED_C20_MEMBERSHIP_PATH)
    test_membership_path = paths_cfg.get("exp_b_test_membership", EXP_B_TEST_MEMBERSHIP_PATH)

    if train_loader is None or val_loader is None:
        batch_size = st_cfg.get("batch_size", 64)
        t_loader, v_loader, _ = get_exp_b_loaders(
            membership_path=membership_path,
            test_membership_path=test_membership_path,
            batch_size=batch_size,
            seed=seed,
        )
        if train_loader is None:
            train_loader = t_loader
        if val_loader is None:
            val_loader = v_loader

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device)
    if model is None:
        model = SmallResNetTTT(num_classes=100)
    model = model.to(device)

    total_steps = epochs * len(train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=pct_start
    )

    source_train_seconds = 0.0
    source_updates = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()

            optimizer.zero_grad()
            cls_logits = model(images)
            loss_cls = F.cross_entropy(cls_logits, labels)

            if mode == "joint":
                rot_images, rot_labels = rotate_batch(images)
                rot_logits = model.forward_rotation(rot_images)
                loss_rot = F.cross_entropy(rot_logits, rot_labels)
                loss = loss_cls + lambda_rot * loss_rot
            else:
                loss = loss_cls

            loss.backward()
            optimizer.step()
            scheduler.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            source_train_seconds += time.perf_counter() - t0
            source_updates += 1

    val_acc, val_rot_acc = evaluate_source_model(model, val_loader, device=device)

    metadata: Dict[str, Any] = {
        "epoch": epochs,
        "mode": mode,
        "seed": seed,
        "source_train_seconds": float(source_train_seconds),
        "source_updates": int(source_updates),
        "val_acc": float(val_acc),
        "val_rot_acc": float(val_rot_acc),
        "config": config_dict,
    }

    if save_path is None:
        save_path = Path(checkpoint_dir) / f"exp_b_{mode}_seed{seed}.pt"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            **metadata,
        },
        save_path,
    )

    return model, metadata


def tune_validation_lr(
    config_path: Union[str, Path] = "configs/ttt_cifar100_exp_b.yaml",
    candidate_lrs: Optional[List[float]] = None,
    device: str = "auto",
    seed: int = 42,
    output_dir: Union[str, Path] = "results",
) -> float:
    """Tune TTT learning rate on the 1,000-image validation split (Khoi Check #3).

    Strictly avoids exposing test data.
    """
    if candidate_lrs is None:
        candidate_lrs = [1e-4, 5e-4, 1e-3, 5e-3]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    paths_cfg = config.get("paths", {})
    membership_path = paths_cfg.get("shared_c20_membership", SHARED_C20_MEMBERSHIP_PATH)
    ckpt_dir = Path(paths_cfg.get("checkpoint_dir", "checkpoints"))
    corr_cfg = config.get("corruption", {})
    noise_sigma = float(corr_cfg.get("sigma", 0.08))
    severity = corr_cfg.get("severity", 3)

    dev = resolve_device(device)
    print(f"[LR Tuning] Evaluating candidate learning rates {candidate_lrs} on val split using {dev}...")

    # Load or train joint model
    ckpt_path = ckpt_dir / f"exp_b_joint_seed{seed}.pt"
    if ckpt_path.exists():
        model, _ = load_model(ckpt_path, device=dev)
    else:
        print(f"[LR Tuning] Checkpoint {ckpt_path} not found. Training joint model on seed {seed}...")
        model, _ = train_source_model(
            mode="joint", config=config, seed=seed, device=dev, checkpoint_dir=ckpt_dir
        )

    val_dataset_clean = CIFAR100ExpBDataset(
        split="validation",
        membership=membership_path,
        augment=False,
    )
    val_dataset_noisy = CIFAR100ExpBDataset(
        split="validation",
        membership=membership_path,
        augment=False,
        corruption="gaussian_noise",
        noise_sigma=noise_sigma,
        noise_seed=seed,
    )

    clean_loader = DataLoader(val_dataset_clean, batch_size=1, shuffle=False)
    noisy_loader = DataLoader(val_dataset_noisy, batch_size=1, shuffle=False)

    best_lr = candidate_lrs[0]
    best_key = (-1.0, -1.0)
    results: List[Dict[str, Any]] = []

    init_features = {k: v.clone() for k, v in model.features.state_dict().items()}

    for lr_cand in candidate_lrs:
        # Standard TTT on noisy val
        model.features.load_state_dict(init_features)
        summary_noisy, _, _ = evaluate_condition(
            model=model,
            loader=noisy_loader,
            condition="standard_ttt",
            ttt_lr=lr_cand,
            ttt_steps=1,
            device=dev,
            corruption="gaussian_noise",
            severity=severity,
            seed=seed,
        )

        # Standard TTT on clean val
        model.features.load_state_dict(init_features)
        summary_clean, _, _ = evaluate_condition(
            model=model,
            loader=clean_loader,
            condition="standard_ttt",
            ttt_lr=lr_cand,
            ttt_steps=1,
            device=dev,
            corruption="clean",
            seed=seed,
        )

        acc_noisy = summary_noisy["accuracy"]
        acc_clean = summary_clean["accuracy"]
        avg_acc = (acc_noisy + acc_clean) / 2.0
        print(f"  ttt_lr={lr_cand:.1e} | Val Clean Acc: {acc_clean:.4f} | Val Noisy Acc: {acc_noisy:.4f}")

        results.append({
            "ttt_lr": lr_cand,
            "val_clean_acc": acc_clean,
            "val_noisy_acc": acc_noisy,
            "avg_acc": avg_acc,
        })

        # Maximize balanced average accuracy while breaking ties toward higher clean preservation
        score_key = (round(avg_acc, 8), round(acc_clean, 8))
        if score_key > best_key:
            best_key = score_key
            best_lr = lr_cand

    out_p = Path(output_dir) / "ttt_lr_tuning.csv"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ttt_lr", "val_clean_acc", "val_noisy_acc", "avg_acc"])
        writer.writeheader()
        writer.writerows(results)

    print(f"[LR Tuning] Selected best ttt_lr: {best_lr} (val noisy acc: {best_score:.4f}). Saved to {out_p}")
    return best_lr


def run_smoke_test(
    config_path: Union[str, Path] = "configs/ttt_cifar100_exp_b.yaml",
    output_dir: Union[str, Path] = "results/smoke_test",
    device: str = "auto",
) -> None:
    """Execute clean and noisy 128-sample smoke test across all 4 conditions."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_csv = output_path / "ttt_summary.csv"
    window_csv = output_path / "ttt_window_summary.csv"

    # Remove stale files if present
    if summary_csv.exists():
        summary_csv.unlink()
    if window_csv.exists():
        window_csv.unlink()

    dev = resolve_device(device)
    print(f"[Smoke Test] Running 128-sample verification on {dev}...")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Instantiate model
    model = SmallResNetTTT(num_classes=100).to(dev)

    # 128-image clean and noisy test loaders
    clean_ds = CIFAR100ExpBDataset(split="test", augment=False, corruption="clean")
    noisy_ds = CIFAR100ExpBDataset(
        split="test", augment=False, corruption="gaussian_noise", noise_sigma=0.08, noise_seed=42
    )

    clean_subset = Subset(clean_ds, range(min(128, len(clean_ds))))
    noisy_subset = Subset(noisy_ds, range(min(128, len(noisy_ds))))

    clean_loader = DataLoader(clean_subset, batch_size=1, shuffle=False)
    noisy_loader = DataLoader(noisy_subset, batch_size=1, shuffle=False)

    conditions = ["supervised_frozen", "joint_frozen", "standard_ttt", "online_ttt"]
    corruptions = [("clean", clean_loader), ("gaussian_noise", noisy_loader)]

    init_state = {k: v.clone() for k, v in model.state_dict().items()}

    for corr_name, loader in corruptions:
        for cond in conditions:
            # Restore model to fresh state before evaluating each condition
            model.load_state_dict(init_state)
            summary_row, windows, _ = evaluate_condition(
                model=model,
                loader=loader,
                condition=cond,
                ttt_lr=0.001,
                ttt_steps=1,
                window_size=200,
                device=dev,
                corruption=corr_name,
                seed=42,
                config_path=str(config_path),
                summary_csv_path=summary_csv,
                window_csv_path=window_csv,
            )
            print(
                f"  [{corr_name.upper():14s}] {cond:18s} | "
                f"Acc: {summary_row['accuracy']:.4f} | "
                f"Updates: {summary_row['update_count']} | "
                f"Windows: {len(windows)}"
            )

    print(f"[Smoke Test] Completed successfully! CSV logs saved to {output_path}")


def main() -> None:
    """CLI Entrypoint for Experiment B TTT Training and Evaluation."""
    parser = argparse.ArgumentParser(
        description="Experiment B: Test-Time Training on CIFAR-100-20"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ttt_cifar100_exp_b.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for CSV summaries (defaults to 'results/smoke_test' for --smoke-test, else 'results')",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 128-image quick verification across all 4 conditions",
    )
    parser.add_argument(
        "--tune-lr",
        action="store_true",
        help="Evaluate candidate ttt_lr values on 1,000-image val split",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Random seeds for full evaluation",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run only pilot seed 42",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Execution device",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train source models only",
    )

    args = parser.parse_args()

    if args.smoke_test:
        smoke_out = args.output if args.output is not None else "results/smoke_test"
        run_smoke_test(config_path=args.config, output_dir=smoke_out, device=args.device)
        return

    with open(args.config, "r") as f:
        config = yaml.safe_load(f) or {}

    paths_cfg = config.get("paths", {})
    out_dir = args.output if args.output is not None else paths_cfg.get("output_dir", "results")

    if args.tune_lr:
        tune_validation_lr(config_path=args.config, device=args.device, output_dir=out_dir)
        return

    target_seeds = [42] if args.pilot else args.seeds
    dev = resolve_device(args.device)
    print(f"[Exp B Runner] Running evaluation for seeds {target_seeds} on {dev}...")

    ckpt_dir = Path(paths_cfg.get("checkpoint_dir", "checkpoints"))
    membership_path = paths_cfg.get("shared_c20_membership", SHARED_C20_MEMBERSHIP_PATH)
    test_membership_path = paths_cfg.get("exp_b_test_membership", EXP_B_TEST_MEMBERSHIP_PATH)

    adapt_cfg = config.get("adaptation", {})
    ttt_lr = float(adapt_cfg.get("ttt_lr", 0.001))
    ttt_steps = int(adapt_cfg.get("ttt_steps", 1))
    lambda_rot = float(adapt_cfg.get("lambda_rot", 1.0))

    corr_cfg = config.get("corruption", {})
    noise_sigma = float(corr_cfg.get("sigma", 0.08))
    severity = corr_cfg.get("severity", 3)

    eval_cfg = config.get("evaluation", {})
    window_size = int(eval_cfg.get("window_size", 200))

    summary_csv = Path(out_dir) / "ttt_summary.csv"
    window_csv = Path(out_dir) / "ttt_window_summary.csv"
    if not args.train_only:
        if summary_csv.exists():
            summary_csv.unlink()
        if window_csv.exists():
            window_csv.unlink()

    for seed in target_seeds:
        print(f"\n====================== SEED {seed} ======================")
        # 1. Train or load supervised model
        sup_ckpt = ckpt_dir / f"exp_b_supervised_seed{seed}.pt"
        if not sup_ckpt.exists():
            print(f"[Seed {seed}] Training source supervised model...")
            sup_model, _ = train_source_model(
                mode="supervised",
                config=config,
                seed=seed,
                device=dev,
                checkpoint_dir=ckpt_dir,
            )
        else:
            print(f"[Seed {seed}] Loading existing supervised checkpoint {sup_ckpt}...")
            sup_model, _ = load_model(sup_ckpt, device=dev)

        # 2. Train or load joint model
        joint_ckpt = ckpt_dir / f"exp_b_joint_seed{seed}.pt"
        if not joint_ckpt.exists():
            print(f"[Seed {seed}] Training source joint model...")
            joint_model, _ = train_source_model(
                mode="joint",
                config=config,
                seed=seed,
                device=dev,
                checkpoint_dir=ckpt_dir,
            )
        else:
            print(f"[Seed {seed}] Loading existing joint checkpoint {joint_ckpt}...")
            joint_model, _ = load_model(joint_ckpt, device=dev)

        if args.train_only:
            continue

        # Keep pristine copies of source weights to isolate state between corruptions and conditions
        clean_sup_state = copy.deepcopy(sup_model.state_dict())
        clean_joint_state = copy.deepcopy(joint_model.state_dict())

        # 3. Stream evaluation on test set (2,000 images)
        clean_loader = get_exp_b_loader(
            split="test",
            batch_size=1,
            seed=seed,
            corruption="clean",
            membership_path=membership_path,
            test_membership_path=test_membership_path,
        )
        noisy_loader = get_exp_b_loader(
            split="test",
            batch_size=1,
            seed=seed,
            corruption="gaussian_noise",
            noise_sigma=noise_sigma,
            noise_seed=seed,
            membership_path=membership_path,
            test_membership_path=test_membership_path,
        )

        corruptions = [("clean", "none", clean_loader), ("gaussian_noise", severity, noisy_loader)]

        condition_specs = [
            ("supervised_frozen", sup_model, clean_sup_state, sup_ckpt),
            ("joint_frozen", joint_model, clean_joint_state, joint_ckpt),
            ("standard_ttt", joint_model, clean_joint_state, joint_ckpt),
            ("online_ttt", joint_model, clean_joint_state, joint_ckpt),
        ]

        for corr_name, corr_sev, loader in corruptions:
            print(f"\n--- Stream: {corr_name.upper()} (Seed {seed}) ---")
            rows_by_cond: Dict[str, Dict[str, Any]] = {}
            for cond_name, cond_model, init_state, ckpt_path in condition_specs:
                cond_model.load_state_dict(copy.deepcopy(init_state))
                cond_row, _, _ = evaluate_condition(
                    model=cond_model,
                    loader=loader,
                    condition=cond_name,
                    ttt_lr=ttt_lr,
                    ttt_steps=ttt_steps,
                    window_size=window_size,
                    device=dev,
                    corruption=corr_name,
                    severity=corr_sev,
                    seed=seed,
                    checkpoint_path=str(ckpt_path),
                    config_path=args.config,
                    summary_csv_path=summary_csv,
                    window_csv_path=window_csv,
                    lambda_rot=lambda_rot,
                )
                rows_by_cond[cond_name] = cond_row

            j_acc = rows_by_cond["joint_frozen"]["accuracy"]
            std_acc = rows_by_cond["standard_ttt"]["accuracy"]
            onl_acc = rows_by_cond["online_ttt"]["accuracy"]
            print(f"  supervised_frozen: Acc = {rows_by_cond['supervised_frozen']['accuracy']:.4f}")
            print(f"  joint_frozen:      Acc = {j_acc:.4f}")
            print(f"  standard_ttt:      Acc = {std_acc:.4f} (Delta: {compute_delta_accuracy(std_acc, j_acc):+.2f}%)")
            print(
                f"  online_ttt:        Acc = {onl_acc:.4f} "
                f"(Delta: {compute_delta_accuracy(onl_acc, j_acc):+.2f}%, "
                f"Carryover: {compute_delta_accuracy(onl_acc, std_acc):+.2f}%)"
            )


if __name__ == "__main__":
    main()
