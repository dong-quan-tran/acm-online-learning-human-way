"""
Periodic Held-Out Evaluation Runner for Experiment B (CIFAR-100 Classes 0-19).

Implements the Team Assignment protocol (docs/11.14-11.20_assignment.md):
- Load existing seed-42 supervised and joint checkpoints without modifying prior results.
- Evaluate fixed clean and noisy held-out sets before adaptation (0 images).
- Stream adaptation images one by one (clean stream and noisy stream separately,
  each starting from a fresh copy of the joint checkpoint).
- Pause every `eval_interval` (100) adaptation images—and after any final group
  of fewer than 100 images—to evaluate both clean and noisy fixed held-out views.
- Carry adapted model and optimizer state forward across evaluation pauses.
- Measure adaptation time separately from held-out evaluation time.
- Enforce and audit Khoi's state-preservation invariants (model weights/GroupNorm/buffers,
  optimizer state, adaptation RNG state, frozen heads, single prediction per image).
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from src.data import (
    EXP_B_TEST_MEMBERSHIP_PATH,
    SHARED_C20_MEMBERSHIP_PATH,
    CIFAR100SplitSubsetDataset,
    load_exp_b_split_pair,
)
from src.models import SmallResNetTTT, rotate_batch
from src.ttt_cifar100 import (
    _cfg_or,
    _cfg_path_or,
    _cfg_section,
    _resolve_model_arch,
    _validate_adapt_and_corruption_cfg,
    freeze_heads_for_ttt,
    load_model,
    resolve_device,
    run_rotation_ttt_steps,
    sync_device,
)

DEFAULT_HELD_OUT_CONFIG = "configs/ttt_cifar100_exp_b_held_out.yaml"
PROTECTED_LEGACY_OUTPUTS = {
    "ttt_summary.csv",
    "ttt_window_summary.csv",
    "ttt_lr_tuning.csv",
}
VALID_HELD_OUT_CONDITIONS = ("supervised_frozen", "joint_frozen", "online_ttt")
VALID_ADAPTATION_STREAMS = ("clean", "noisy")
VALID_HELD_OUT_VIEWS = ("clean", "noisy")
_UNSET = object()

HELD_OUT_CSV_COLUMNS: Tuple[str, ...] = (
    "experiment",
    "condition",
    "seed",
    "source_checkpoint",
    "code_version",
    "adaptation_stream_condition",
    "held_out_condition",
    "adaptation_images_processed",
    "held_out_image_count",
    "correct_prediction_count",
    "held_out_accuracy",
    "adaptation_update_count",
    "adaptation_time_seconds",
    "adaptation_interval_time_seconds",
    "held_out_eval_time_seconds",
    "cumulative_held_out_eval_time_seconds",
    "config_path",
    "split_locations",
    "hardware",
)


@dataclass
class HeldOutEvalResult:
    """Result of one fixed held-out evaluation pass."""

    image_count: int
    correct_count: int
    accuracy: float
    eval_time_seconds: float
    predictions: List[int]
    sample_ids: List[str]


def get_code_version_strict() -> str:
    """Return short git SHA, appending '-dirty' if working tree has local changes."""
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "git-head"


def _recursive_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _recursive_equal(left[k], right[k]) for k in left
        )
    if isinstance(left, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_recursive_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def _clone_nested(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, dict):
        return {k: _clone_nested(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clone_nested(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_nested(v) for v in obj)
    return copy.deepcopy(obj)


def _capture_rng_state(device: torch.device) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": _clone_nested(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        state["torch_cuda"] = [s.clone() for s in torch.cuda.get_rng_state_all()]
    if (
        device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
        and torch.backends.mps.is_available()
    ):
        mps_state = torch.mps.get_rng_state()
        state["torch_mps"] = (
            mps_state.detach().cpu().clone()
            if isinstance(mps_state, torch.Tensor)
            else _clone_nested(mps_state)
        )
    return state


def _assert_rng_state_unchanged(before: Dict[str, Any], device: torch.device) -> None:
    after = _capture_rng_state(device)
    if not _recursive_equal(before, after):
        raise RuntimeError(
            "Invariant violated: held-out evaluation modified random-number state"
        )


def _capture_model_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _assert_model_state_unchanged(
    before: Dict[str, torch.Tensor], model: torch.nn.Module
) -> None:
    after = model.state_dict()
    if set(before.keys()) != set(after.keys()):
        raise RuntimeError("Invariant violated: model state_dict keys changed during eval")
    for k, v_before in before.items():
        if not torch.equal(v_before, after[k].detach().cpu()):
            raise RuntimeError(
                f"Invariant violated: model parameter/buffer '{k}' changed during held-out eval"
            )


def _assert_optimizer_state_unchanged(
    before_sd: Optional[Dict[str, Any]],
    optimizer: Optional[torch.optim.Optimizer],
) -> None:
    if optimizer is None or before_sd is None:
        return
    after_sd = _clone_nested(optimizer.state_dict())
    if not _recursive_equal(before_sd, after_sd):
        raise RuntimeError(
            "Invariant violated: optimizer state changed during held-out evaluation"
        )


def _validate_feature_optimizer(
    model: SmallResNetTTT,
    optimizer: torch.optim.Optimizer,
) -> None:
    feature_param_ids = {id(p) for p in model.features.parameters()}
    opt_param_ids = {
        id(p) for grp in optimizer.param_groups for p in grp["params"]
    }
    if opt_param_ids != feature_param_ids:
        raise ValueError(
            "Adaptation optimizer must bind strictly and completely to model.features.parameters()"
        )


def _adapt_image_step(
    model: SmallResNetTTT,
    image: torch.Tensor,
    *,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    ttt_lr: float,
    ttt_steps: int,
    lambda_rot: float,
) -> float:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4 or image.size(0) != 1:
        raise ValueError(
            f"adapt_image_only expects 3D or (1, C, H, W) tensor, got {tuple(image.shape)}"
        )

    image = image.to(device)
    rot_images, rot_labels = rotate_batch(image)

    if ttt_lr > 0.0 and ttt_steps > 0:
        active_opt = optimizer or torch.optim.SGD(
            model.features.parameters(), lr=ttt_lr, momentum=0.0
        )
        return run_rotation_ttt_steps(
            model,
            rot_images,
            rot_labels,
            optimizer=active_opt,
            ttt_steps=ttt_steps,
            lambda_rot=lambda_rot,
        )

    with torch.no_grad():
        rot_logits = model.forward_rotation(rot_images)
        return float((lambda_rot * F.cross_entropy(rot_logits, rot_labels)).item())


def adapt_image_only(
    model: SmallResNetTTT,
    image: torch.Tensor,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ttt_lr: float = 0.001,
    ttt_steps: int = 1,
    lambda_rot: float = 1.0,
) -> float:
    """Perform label-free rotation TTT updates on one image using persistent optimizer state.

    Accepts ONLY image tensors (no class-label parameter). Freezes classifier and
    rotation_head and updates model.features on 4 rotated views [0, 90, 180, 270].
    """
    if ttt_lr < 0.0:
        raise ValueError(f"ttt_lr must be non-negative, got {ttt_lr}")
    if ttt_steps < 0:
        raise ValueError(f"ttt_steps must be non-negative, got {ttt_steps}")

    freeze_heads_for_ttt(model)
    if optimizer is not None:
        _validate_feature_optimizer(model, optimizer)

    device = next(model.parameters()).device
    last_loss = _adapt_image_step(
        model,
        image,
        device=device,
        optimizer=optimizer,
        ttt_lr=ttt_lr,
        ttt_steps=ttt_steps,
        lambda_rot=lambda_rot,
    )
    freeze_heads_for_ttt(model)
    return last_loss


def prepare_held_out_batches(
    source: Union[DataLoader, Dataset, Sequence[Tuple[torch.Tensor, torch.Tensor, Sequence[str]]]],
    batch_size: int = 64,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor, Tuple[str, ...]]], Tuple[str, ...]]:
    """Materialize fixed held-out batches once so inputs never vary across checkpoints."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if isinstance(source, Dataset):
        loader: Any = DataLoader(source, batch_size=batch_size, shuffle=False, drop_last=False)
    else:
        loader = source

    all_images: List[torch.Tensor] = []
    all_labels: List[int] = []
    all_ids: List[str] = []

    for batch in loader:
        images, labels, sample_ids = batch
        if isinstance(labels, torch.Tensor):
            lbl_list = [int(x) for x in labels.view(-1).tolist()]
        elif isinstance(labels, (list, tuple)):
            lbl_list = [int(x) for x in labels]
        else:
            lbl_list = [int(labels)]

        if isinstance(sample_ids, torch.Tensor):
            sid_list = [str(x) for x in sample_ids.view(-1).tolist()]
        elif isinstance(sample_ids, (list, tuple)):
            sid_list = [str(s) for s in sample_ids]
        else:
            sid_list = [str(sample_ids)]

        if images.dim() == 3:
            images = images.unsqueeze(0)
        if len(sid_list) != images.size(0) or len(lbl_list) != images.size(0):
            raise ValueError("Batch mismatch between images, labels, and sample_ids")

        for idx in range(images.size(0)):
            all_images.append(images[idx].detach().cpu().clone())
            all_labels.append(lbl_list[idx])
            all_ids.append(sid_list[idx])

    if not all_ids:
        raise ValueError("Held-out loader is empty")
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Held-out loader contains duplicate sample IDs")

    batches: List[Tuple[torch.Tensor, torch.Tensor, Tuple[str, ...]]] = []
    for start in range(0, len(all_ids), batch_size):
        end = min(start + batch_size, len(all_ids))
        batch_imgs = torch.stack(all_images[start:end], dim=0)
        batch_lbls = torch.tensor(all_labels[start:end], dtype=torch.long)
        batch_sids = tuple(all_ids[start:end])
        batches.append((batch_imgs, batch_lbls, batch_sids))

    return batches, tuple(all_ids)


def evaluate_held_out_set(
    model: SmallResNetTTT,
    held_out_batches: Sequence[Tuple[torch.Tensor, torch.Tensor, Sequence[str]]],
    expected_sample_ids: Sequence[str],
    device: Union[str, torch.device] = "cpu",
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    verify_invariants: bool = True,
    expected_num_outputs: int = 100,
) -> HeldOutEvalResult:
    """Evaluate fixed held-out batches under torch.no_grad() and model.eval().

    When ``verify_invariants`` is True, verifies outside the timing window that
    model parameters/buffers, optimizer state, and RNG state remain identical and
    that every expected held-out image receives exactly one prediction.
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    before_model = _capture_model_state(model) if verify_invariants else None
    before_opt = _clone_nested(optimizer.state_dict()) if (verify_invariants and optimizer is not None) else None
    before_rng = _capture_rng_state(dev) if verify_invariants else None

    predictions: List[int] = []
    recorded_ids: List[str] = []
    correct_count = 0

    sync_device(dev)
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch_imgs, batch_lbls, batch_sids in held_out_batches:
            imgs = batch_imgs.to(dev)
            logits = model(imgs)
            if logits.dim() != 2 or logits.size(1) != expected_num_outputs:
                raise RuntimeError(
                    f"Expected unmasked (B, {expected_num_outputs}) logits, "
                    f"got shape {tuple(logits.shape)}"
                )
            preds = logits.argmax(dim=-1).cpu().tolist()
            lbls = batch_lbls.tolist()
            for pred, lbl, sid in zip(preds, lbls, batch_sids):
                pred_int = int(pred)
                if not 0 <= pred_int < expected_num_outputs:
                    raise RuntimeError(
                        f"Prediction {pred_int} outside 0..{expected_num_outputs - 1} range"
                    )
                predictions.append(pred_int)
                recorded_ids.append(str(sid))
                if pred_int == int(lbl):
                    correct_count += 1
    sync_device(dev)
    eval_time = time.perf_counter() - t0

    if recorded_ids != list(expected_sample_ids):
        raise RuntimeError(
            "Invariant violated: recorded held-out predictions do not match expected IDs/order 1-to-1"
        )
    if model.training:
        raise RuntimeError("Invariant violated: model entered training mode during held-out eval")

    if verify_invariants:
        assert before_model is not None and before_rng is not None
        _assert_model_state_unchanged(before_model, model)
        _assert_optimizer_state_unchanged(before_opt, optimizer)
        _assert_rng_state_unchanged(before_rng, dev)

    total = len(predictions)
    acc = float(correct_count) / total
    return HeldOutEvalResult(
        image_count=total,
        correct_count=correct_count,
        accuracy=acc,
        eval_time_seconds=float(eval_time),
        predictions=predictions,
        sample_ids=recorded_ids,
    )


def build_held_out_row(
    *,
    experiment: str = "B",
    condition: str,
    seed: int,
    source_checkpoint: str,
    code_version: str,
    adaptation_stream_condition: str,
    held_out_condition: str,
    adaptation_images_processed: int,
    held_out_image_count: int,
    correct_prediction_count: int,
    held_out_accuracy: float,
    adaptation_update_count: int,
    adaptation_time_seconds: float,
    held_out_eval_time_seconds: float,
    config_path: str,
    split_locations: str,
    hardware: str,
    adaptation_interval_time_seconds: float = 0.0,
    cumulative_held_out_eval_time_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a validated CSV dictionary matching Section-4 fields and explicit timing columns."""
    if held_out_condition not in {"clean", "noisy"}:
        raise ValueError(f"held_out_condition must be 'clean' or 'noisy', got {held_out_condition!r}")
    if adaptation_stream_condition not in {"none", "clean", "noisy"}:
        raise ValueError(
            f"adaptation_stream_condition must be 'none', 'clean', or 'noisy', got {adaptation_stream_condition!r}"
        )
    if condition not in {"supervised_frozen", "joint_frozen", "online_ttt"}:
        raise ValueError(f"Unsupported condition {condition!r}")
    if not 0.0 <= held_out_accuracy <= 1.0:
        raise ValueError(f"held_out_accuracy out of bounds: {held_out_accuracy}")
    if correct_prediction_count < 0 or correct_prediction_count > held_out_image_count or held_out_image_count <= 0:
        raise ValueError("Invalid held_out_image_count / correct_prediction_count")
    if (
        adaptation_images_processed < 0
        or adaptation_update_count < 0
        or adaptation_time_seconds < 0.0
        or adaptation_interval_time_seconds < 0.0
        or held_out_eval_time_seconds < 0.0
    ):
        raise ValueError("Counts and timing values must be non-negative")

    cum_eval_time = (
        float(held_out_eval_time_seconds)
        if cumulative_held_out_eval_time_seconds is None
        else float(cumulative_held_out_eval_time_seconds)
    )
    if cum_eval_time < 0.0:
        raise ValueError("cumulative_held_out_eval_time_seconds must be non-negative")

    return {
        "experiment": str(experiment),
        "condition": str(condition),
        "seed": int(seed),
        "source_checkpoint": str(source_checkpoint),
        "code_version": str(code_version),
        "adaptation_stream_condition": str(adaptation_stream_condition),
        "held_out_condition": str(held_out_condition),
        "adaptation_images_processed": int(adaptation_images_processed),
        "held_out_image_count": int(held_out_image_count),
        "correct_prediction_count": int(correct_prediction_count),
        "held_out_accuracy": float(held_out_accuracy),
        "adaptation_update_count": int(adaptation_update_count),
        "adaptation_time_seconds": float(adaptation_time_seconds),
        "adaptation_interval_time_seconds": float(adaptation_interval_time_seconds),
        "held_out_eval_time_seconds": float(held_out_eval_time_seconds),
        "cumulative_held_out_eval_time_seconds": cum_eval_time,
        "config_path": str(config_path),
        "split_locations": str(split_locations),
        "hardware": str(hardware),
    }


def _iter_adaptation_images(
    adaptation_loader: Optional[Union[DataLoader, Sequence[Any]]],
    max_adapt_images: Optional[int] = None,
) -> Iterator[Tuple[torch.Tensor, str]]:
    """Yield ``(image_1xCxHxW, sample_id)`` lazily while keeping class labels outside adaptation."""
    if adaptation_loader is None:
        return
    if max_adapt_images is not None and max_adapt_images <= 0:
        raise ValueError(f"max_adapt_images must be positive, got {max_adapt_images}")

    yielded = 0
    for batch in adaptation_loader:
        if max_adapt_images is not None and yielded >= max_adapt_images:
            return
        if isinstance(batch, torch.Tensor):
            imgs = batch
            has_sids = False
            raw_sids: Any = None
        elif isinstance(batch, (list, tuple)):
            if len(batch) == 0:
                raise ValueError("Adaptation batch cannot be empty")
            imgs = batch[0]
            has_sids = len(batch) >= 3 and batch[2] is not None
            raw_sids = batch[2] if has_sids else None
        else:
            raise TypeError(f"Unsupported adaptation batch type: {type(batch).__name__}")

        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)
        if imgs.dim() != 4:
            raise ValueError(
                f"Expected 3D or 4D adaptation image tensor, got shape {tuple(imgs.shape)}"
            )
        batch_n = imgs.size(0)
        if batch_n == 0:
            continue

        if has_sids:
            if isinstance(raw_sids, torch.Tensor):
                sids = [str(x) for x in raw_sids.view(-1).tolist()]
            elif isinstance(raw_sids, (list, tuple)):
                sids = [str(x) for x in raw_sids]
            else:
                sids = [str(raw_sids)]
            if len(sids) != batch_n:
                raise ValueError(
                    f"Adaptation batch size mismatch: {batch_n} images vs {len(sids)} sample_ids"
                )
        else:
            sids = [f"adapt:{yielded + i}" for i in range(batch_n)]

        for i in range(batch_n):
            yield imgs[i : i + 1], sids[i]
            yielded += 1
            if max_adapt_images is not None and yielded >= max_adapt_images:
                return


def _validate_selection_list(
    values: Sequence[str],
    allowed: Sequence[str],
    name: str,
    *,
    allow_empty: bool = False,
) -> Tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple, got {type(values).__name__}")
    items = [str(v) for v in values]
    if not allow_empty and not items:
        raise ValueError(f"{name} must be non-empty")
    unknown = [v for v in items if v not in allowed]
    if unknown:
        raise ValueError(f"Unsupported {name} value(s): {unknown}; allowed: {list(allowed)}")
    if len(set(items)) != len(items):
        raise ValueError(f"{name} contains duplicate values: {items}")
    return tuple(items)


def run_held_out_stream(
    *,
    model: SmallResNetTTT,
    adaptation_loader: Optional[Union[DataLoader, Sequence[Any]]],
    held_out_clean_batches: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor, Sequence[str]]]] = None,
    held_out_clean_ids: Optional[Sequence[str]] = None,
    held_out_noisy_batches: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor, Sequence[str]]]] = None,
    held_out_noisy_ids: Optional[Sequence[str]] = None,
    held_out_views: Sequence[str] = VALID_HELD_OUT_VIEWS,
    condition: str,
    adaptation_stream_condition: str,
    eval_interval: int = 100,
    ttt_lr: float = 0.001,
    ttt_steps: int = 1,
    ttt_momentum: float = 0.0,
    lambda_rot: float = 1.0,
    seed: int = 42,
    source_checkpoint: str = "",
    config_path: str = DEFAULT_HELD_OUT_CONFIG,
    split_locations: str = "",
    code_version: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
    verify_invariants: bool = True,
    max_adapt_images: Optional[int] = None,
    expected_num_outputs: int = 100,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run one frozen reference condition or one online TTT adaptation stream."""
    if eval_interval <= 0:
        raise ValueError(f"eval_interval must be positive, got {eval_interval}")
    if ttt_lr < 0.0:
        raise ValueError(f"ttt_lr must be non-negative, got {ttt_lr}")
    if ttt_steps < 0:
        raise ValueError(f"ttt_steps must be non-negative, got {ttt_steps}")
    if condition not in VALID_HELD_OUT_CONDITIONS:
        raise ValueError(f"Unsupported condition {condition!r}")

    active_views = _validate_selection_list(
        held_out_views, VALID_HELD_OUT_VIEWS, "held_out_views"
    )
    view_payloads: List[
        Tuple[
            str,
            Sequence[Tuple[torch.Tensor, torch.Tensor, Sequence[str]]],
            Sequence[str],
        ]
    ] = []
    for view in active_views:
        if view == "clean":
            if held_out_clean_batches is None or held_out_clean_ids is None:
                raise ValueError("held_out_clean_batches and held_out_clean_ids are required for 'clean' view")
            view_payloads.append(("clean", held_out_clean_batches, held_out_clean_ids))
        else:
            if held_out_noisy_batches is None or held_out_noisy_ids is None:
                raise ValueError("held_out_noisy_batches and held_out_noisy_ids are required for 'noisy' view")
            view_payloads.append(("noisy", held_out_noisy_batches, held_out_noisy_ids))

    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    ver = code_version or get_code_version_strict()

    rows: List[Dict[str, Any]] = []
    recorded_checkpoints: List[int] = []
    eval_time_cum_by_view: Dict[str, float] = {view: 0.0 for view in active_views}

    def _record_checkpoint(
        processed: int,
        updates: int,
        adapt_time_cum: float,
        adapt_time_interval: float,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> None:
        recorded_checkpoints.append(processed)
        for ho_cond, ho_batches, ho_ids in view_payloads:
            res = evaluate_held_out_set(
                model,
                ho_batches,
                ho_ids,
                device=dev,
                optimizer=optimizer,
                verify_invariants=verify_invariants,
                expected_num_outputs=expected_num_outputs,
            )
            eval_time_cum_by_view[ho_cond] += res.eval_time_seconds
            rows.append(
                build_held_out_row(
                    experiment="B",
                    condition=condition,
                    seed=seed,
                    source_checkpoint=source_checkpoint,
                    code_version=ver,
                    adaptation_stream_condition=adaptation_stream_condition,
                    held_out_condition=ho_cond,
                    adaptation_images_processed=processed,
                    held_out_image_count=res.image_count,
                    correct_prediction_count=res.correct_count,
                    held_out_accuracy=res.accuracy,
                    adaptation_update_count=updates,
                    adaptation_time_seconds=adapt_time_cum,
                    adaptation_interval_time_seconds=adapt_time_interval,
                    held_out_eval_time_seconds=res.eval_time_seconds,
                    cumulative_held_out_eval_time_seconds=eval_time_cum_by_view[ho_cond],
                    config_path=config_path,
                    split_locations=split_locations,
                    hardware=str(dev.type),
                )
            )

    if condition in {"supervised_frozen", "joint_frozen"}:
        _record_checkpoint(0, 0, 0.0, 0.0, optimizer=None)
        return rows, {
            "condition": condition,
            "adaptation_stream_condition": adaptation_stream_condition,
            "held_out_views": list(active_views),
            "checkpoints_evaluated": recorded_checkpoints,
            "total_adaptation_images": 0,
            "total_updates": 0,
            "total_adaptation_time_seconds": 0.0,
            "total_held_out_eval_time_by_view": dict(eval_time_cum_by_view),
            "total_held_out_eval_time_seconds": sum(eval_time_cum_by_view.values()),
            "all_invariants_passed": True,
        }

    # Online TTT: stream images lazily (O(1) memory, labels never enter adaptation)
    adapt_iter = _iter_adaptation_images(adaptation_loader, max_adapt_images=max_adapt_images)
    try:
        first_item = next(adapt_iter)
    except StopIteration as exc:
        raise ValueError("Adaptation stream is empty for online_ttt") from exc

    freeze_heads_for_ttt(model)

    init_cls_sd = _capture_model_state(model.classifier)
    init_rot_sd = _capture_model_state(model.rotation_head)
    init_feat_sd = _capture_model_state(model.features)

    optimizer = torch.optim.SGD(
        model.features.parameters(), lr=ttt_lr, momentum=ttt_momentum
    )
    _validate_feature_optimizer(model, optimizer)

    # Checkpoint 0: before any adaptation
    _record_checkpoint(0, 0, 0.0, 0.0, optimizer=optimizer)

    processed = 0
    updates = 0
    adapt_time_cum = 0.0
    performs_updates = ttt_lr > 0.0 and ttt_steps > 0

    sync_device(dev)
    chunk_t0 = time.perf_counter()

    def _flush_checkpoint(n_processed: int) -> None:
        nonlocal adapt_time_cum, chunk_t0
        sync_device(dev)
        interval_time = time.perf_counter() - chunk_t0
        adapt_time_cum += interval_time
        if verify_invariants:
            _assert_model_state_unchanged(init_cls_sd, model.classifier)
            _assert_model_state_unchanged(init_rot_sd, model.rotation_head)
        _record_checkpoint(
            n_processed,
            updates,
            adapt_time_cum,
            interval_time,
            optimizer=optimizer,
        )
        sync_device(dev)
        chunk_t0 = time.perf_counter()

    for img, _sid in itertools.chain([first_item], adapt_iter):
        _adapt_image_step(
            model,
            img,
            device=dev,
            optimizer=optimizer,
            ttt_lr=ttt_lr,
            ttt_steps=ttt_steps,
            lambda_rot=lambda_rot,
        )
        processed += 1
        if performs_updates:
            updates += ttt_steps

        if processed % eval_interval == 0:
            _flush_checkpoint(processed)

    total_adapt = processed
    if total_adapt % eval_interval != 0:
        _flush_checkpoint(total_adapt)

    freeze_heads_for_ttt(model)

    expected_checkpoints = [0] + list(range(eval_interval, total_adapt + 1, eval_interval))
    if expected_checkpoints[-1] != total_adapt:
        expected_checkpoints.append(total_adapt)
    if recorded_checkpoints != expected_checkpoints:
        raise RuntimeError(
            f"Checkpoint schedule mismatch: recorded {recorded_checkpoints}, expected {expected_checkpoints}"
        )

    features_changed = any(
        not torch.equal(init_feat_sd[k], v.detach().cpu())
        for k, v in model.features.state_dict().items()
    )
    if performs_updates and not features_changed:
        raise RuntimeError("Invariant violated: online_ttt did not update model.features")

    return rows, {
        "condition": condition,
        "adaptation_stream_condition": adaptation_stream_condition,
        "held_out_views": list(active_views),
        "checkpoints_evaluated": recorded_checkpoints,
        "total_adaptation_images": total_adapt,
        "total_updates": updates,
        "total_adaptation_time_seconds": adapt_time_cum,
        "total_held_out_eval_time_by_view": dict(eval_time_cum_by_view),
        "total_held_out_eval_time_seconds": sum(eval_time_cum_by_view.values()),
        "frozen_heads_preserved": True,
        "features_adapted": features_changed,
        "all_invariants_passed": True,
    }


def _reject_empty_path(val: Any, message: str) -> None:
    if (isinstance(val, str) and not val.strip()) or (
        isinstance(val, Path) and not val.parts
    ):
        raise ValueError(message)


def load_schema_map(
    schema_map_path: Optional[Union[str, Path, Dict[str, str]]],
) -> Optional[Dict[str, str]]:
    if schema_map_path is None:
        return None
    if isinstance(schema_map_path, dict):
        mapping = schema_map_path
    else:
        _reject_empty_path(schema_map_path, "schema_map_path must be a non-empty path")
        with open(schema_map_path, "r", encoding="utf-8") as handle:
            mapping = json.load(handle)
    if not isinstance(mapping, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()
    ):
        raise ValueError("schema_map must be a JSON object mapping canonical column names to target names")
    unknown = set(mapping.keys()) - set(HELD_OUT_CSV_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown canonical column(s) in schema_map: {sorted(unknown)}")
    return mapping


def save_held_out_csv(
    rows: Sequence[Dict[str, Any]],
    output_path: Union[str, Path],
    schema_map: Optional[Dict[str, str]] = None,
    overwrite: bool = False,
) -> Path:
    """Write held-out evaluation CSV while protecting legacy and existing outputs."""
    out = Path(output_path)
    if out.name in PROTECTED_LEGACY_OUTPUTS:
        raise ValueError(
            f"Refusing to overwrite legacy Experiment B output '{out.name}'. "
            "Use a separate output filename for this week's held-out evaluation."
        )
    if out.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output CSV '{out}'. "
            "Choose a new --output-csv path or pass --overwrite."
        )
    out.parent.mkdir(parents=True, exist_ok=True)

    col_order = [schema_map.get(col, col) if schema_map else col for col in HELD_OUT_CSV_COLUMNS]
    if len(set(col_order)) != len(col_order):
        raise ValueError("schema_map produces duplicate target column names")

    tmp_out = out.with_name(f".{out.name}.tmp.{os.getpid()}")
    try:
        with open(tmp_out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=col_order)
            writer.writeheader()
            for row in rows:
                mapped_row = {
                    (schema_map.get(col, col) if schema_map else col): row.get(col, "")
                    for col in HELD_OUT_CSV_COLUMNS
                }
                writer.writerow(mapped_row)
        os.replace(tmp_out, out)
    finally:
        if tmp_out.exists():
            tmp_out.unlink()
    return out


def _validate_checkpoint_metadata(
    ckpt_meta: Dict[str, Any],
    *,
    ckpt_path: Path,
    condition: str,
    expected_seed: int,
) -> None:
    expected_mode = "supervised" if condition == "supervised_frozen" else "joint"
    if "mode" in ckpt_meta and ckpt_meta["mode"] is not None:
        actual_mode = str(ckpt_meta["mode"])
        if actual_mode != expected_mode:
            raise ValueError(
                f"Checkpoint '{ckpt_path}' has mode={actual_mode!r}, expected {expected_mode!r} "
                f"for condition '{condition}'"
            )
    if "seed" in ckpt_meta and ckpt_meta["seed"] is not None:
        actual_seed = int(ckpt_meta["seed"])
        if actual_seed != int(expected_seed):
            raise ValueError(
                f"Checkpoint '{ckpt_path}' has seed={actual_seed}, expected seed={expected_seed}"
            )


def _load_yaml_mapping(config_path: Union[str, Path]) -> Dict[str, Any]:
    _reject_empty_path(config_path, "config_path must be a non-empty path")
    cfg_path_obj = Path(config_path)
    if not cfg_path_obj.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path_obj}")
    with open(cfg_path_obj, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must parse to a YAML mapping: {cfg_path_obj}")
    return cfg


def run_held_out_experiment(
    *,
    split_file: Optional[Union[str, Path, Dict[str, Any]]] = None,
    held_out_file: Optional[Union[str, Path, Dict[str, Any]]] = None,
    adaptation_key: Optional[str] = None,
    held_out_key: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = DEFAULT_HELD_OUT_CONFIG,
    output_csv: Optional[Union[str, Path]] = None,
    output_checks: Optional[Union[str, Path]] = None,
    supervised_ckpt: Optional[Union[str, Path]] = None,
    joint_ckpt: Optional[Union[str, Path]] = None,
    seed: Optional[int] = None,
    eval_interval: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    ttt_lr: Optional[float] = None,
    ttt_steps: Optional[int] = None,
    ttt_momentum: Optional[float] = None,
    lambda_rot: Optional[float] = None,
    noise_sigma: Optional[float] = None,
    noise_seed: Optional[int] = None,
    conditions: Optional[Sequence[str]] = None,
    adaptation_streams: Optional[Sequence[str]] = None,
    held_out_views: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
    max_adapt_images: Optional[int] = None,
    verify_invariants: bool = True,
    schema_map_path: Optional[Union[str, Path, Dict[str, str]]] = None,
    datasets_by_split: Optional[Dict[str, Any]] = None,
    train_val_membership: Any = _UNSET,
    prior_test_membership: Any = _UNSET,
    overwrite: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run all configured frozen references + online TTT held-out evaluations."""
    cfg: Dict[str, Any] = {}
    config_path_str = "<none>"
    if config_path is not None:
        _reject_empty_path(config_path, "config_path must be a non-empty path")
        config_path_str = str(config_path)
        cfg = _load_yaml_mapping(config_path)

    if split_file is not None:
        _reject_empty_path(split_file, "split_file must be a non-empty path or split mapping")
    if held_out_file is not None:
        _reject_empty_path(
            held_out_file, "held_out_file must be a non-empty path or split mapping"
        )
    if output_csv is not None:
        _reject_empty_path(output_csv, "output_csv must be a non-empty path")
    if output_checks is not None:
        _reject_empty_path(output_checks, "output_checks must be a non-empty path")
    if supervised_ckpt is not None:
        _reject_empty_path(supervised_ckpt, "supervised_ckpt must be a non-empty path")
    if joint_ckpt is not None:
        _reject_empty_path(joint_ckpt, "joint_ckpt must be a non-empty path")

    _resolve_model_arch(cfg)

    adapt_cfg = _cfg_section(cfg, "adaptation")
    corr_cfg = _cfg_section(cfg, "corruption")
    eval_cfg = (
        _cfg_section(cfg, "held_out_evaluation")
        if cfg.get("held_out_evaluation") is not None
        else _cfg_section(cfg, "evaluation")
    )
    paths_cfg = _cfg_section(cfg, "paths")

    ttt_optimizer, _ = _validate_adapt_and_corruption_cfg(adapt_cfg, corr_cfg)

    cfg_seed = eval_cfg.get("seed")
    resolved_seed = int(_cfg_or(eval_cfg, "seed", 42) if seed is None else seed)
    resolved_eval_interval = int(
        _cfg_or(eval_cfg, "eval_interval", 100) if eval_interval is None else eval_interval
    )
    resolved_eval_batch_size = int(
        _cfg_or(eval_cfg, "eval_batch_size", 64) if eval_batch_size is None else eval_batch_size
    )
    resolved_ttt_lr = float(_cfg_or(adapt_cfg, "ttt_lr", 0.001) if ttt_lr is None else ttt_lr)
    resolved_ttt_steps = int(_cfg_or(adapt_cfg, "ttt_steps", 1) if ttt_steps is None else ttt_steps)
    resolved_ttt_momentum = float(
        _cfg_or(adapt_cfg, "ttt_momentum", 0.0) if ttt_momentum is None else ttt_momentum
    )
    resolved_lambda_rot = float(
        _cfg_or(adapt_cfg, "lambda_rot", 1.0) if lambda_rot is None else lambda_rot
    )
    resolved_noise_sigma = float(
        _cfg_or(corr_cfg, "sigma", 0.08) if noise_sigma is None else noise_sigma
    )
    if resolved_noise_sigma < 0.0:
        raise ValueError(f"noise_sigma must be non-negative, got {resolved_noise_sigma}")
    resolved_noise_seed = int(
        _cfg_or(corr_cfg, "noise_seed", 42) if noise_seed is None else noise_seed
    )
    resolved_device = str(_cfg_or(eval_cfg, "device", "auto") if device is None else device)

    raw_conditions = (
        _cfg_or(eval_cfg, "conditions", list(VALID_HELD_OUT_CONDITIONS))
        if conditions is None
        else conditions
    )
    resolved_conditions = _validate_selection_list(
        raw_conditions, VALID_HELD_OUT_CONDITIONS, "held_out_evaluation.conditions"
    )
    raw_streams = (
        _cfg_or(eval_cfg, "adaptation_streams", list(VALID_ADAPTATION_STREAMS))
        if adaptation_streams is None
        else adaptation_streams
    )
    resolved_streams = _validate_selection_list(
        raw_streams,
        VALID_ADAPTATION_STREAMS,
        "held_out_evaluation.adaptation_streams",
        allow_empty=("online_ttt" not in resolved_conditions),
    )
    raw_views = (
        _cfg_or(eval_cfg, "held_out_views", list(VALID_HELD_OUT_VIEWS))
        if held_out_views is None
        else held_out_views
    )
    resolved_views = _validate_selection_list(
        raw_views, VALID_HELD_OUT_VIEWS, "held_out_evaluation.held_out_views"
    )

    resolved_split_file = (
        _cfg_path_or(paths_cfg, "split_file", None) if split_file is None else split_file
    )
    if resolved_split_file is None:
        raise ValueError("split_file must be a non-empty path or split mapping")
    resolved_held_out_file = (
        _cfg_path_or(paths_cfg, "held_out_file", None)
        if held_out_file is None
        else held_out_file
    )

    resolved_output_csv = (
        _cfg_path_or(paths_cfg, "output_csv", "results/exp_b_held_out_pilot.csv")
        if output_csv is None
        else output_csv
    )
    out_path_check = Path(resolved_output_csv)
    if out_path_check.name in PROTECTED_LEGACY_OUTPUTS:
        raise ValueError(
            f"Refusing to overwrite legacy Experiment B output '{out_path_check.name}'. "
            "Use a separate output filename for this week's held-out evaluation."
        )
    if out_path_check.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output CSV '{out_path_check}'. "
            "Choose a new --output-csv path or pass --overwrite."
        )

    resolved_output_checks = (
        _cfg_path_or(paths_cfg, "output_checks", None)
        if output_checks is None
        else output_checks
    )
    checks_path = (
        Path(resolved_output_checks)
        if resolved_output_checks is not None
        else out_path_check.with_name(f"{out_path_check.stem}_checks.json")
    )
    if checks_path.name in PROTECTED_LEGACY_OUTPUTS:
        raise ValueError(
            f"Refusing to overwrite legacy Experiment B output '{checks_path.name}'."
        )
    if checks_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing checks JSON '{checks_path}'. "
            "Choose a new output path or pass --overwrite."
        )

    resolved_train_val = (
        _cfg_path_or(paths_cfg, "shared_c20_membership", SHARED_C20_MEMBERSHIP_PATH)
        if train_val_membership is _UNSET
        else train_val_membership
    )
    resolved_prior_test = (
        _cfg_path_or(paths_cfg, "prior_exp_b_test_membership", EXP_B_TEST_MEMBERSHIP_PATH)
        if prior_test_membership is _UNSET
        else prior_test_membership
    )
    if resolved_train_val is not None:
        _reject_empty_path(
            resolved_train_val,
            "train_val_membership must be a non-empty path, mapping, or None",
        )
    if resolved_prior_test is not None:
        _reject_empty_path(
            resolved_prior_test,
            "prior_test_membership must be a non-empty path, mapping, or None",
        )

    ckpt_dir = Path(_cfg_path_or(paths_cfg, "checkpoint_dir", "checkpoints"))
    use_cfg_ckpt_paths = cfg_seed is None or resolved_seed == int(cfg_seed)
    cfg_sup_ckpt = (
        _cfg_path_or(paths_cfg, "supervised_checkpoint", None) if use_cfg_ckpt_paths else None
    )
    cfg_jnt_ckpt = (
        _cfg_path_or(paths_cfg, "joint_checkpoint", None) if use_cfg_ckpt_paths else None
    )

    sup_path = (
        Path(supervised_ckpt)
        if supervised_ckpt is not None
        else (Path(cfg_sup_ckpt) if cfg_sup_ckpt else ckpt_dir / f"exp_b_supervised_seed{resolved_seed}.pt")
    )
    jnt_path = (
        Path(joint_ckpt)
        if joint_ckpt is not None
        else (Path(cfg_jnt_ckpt) if cfg_jnt_ckpt else ckpt_dir / f"exp_b_joint_seed{resolved_seed}.pt")
    )
    if "supervised_frozen" in resolved_conditions and not sup_path.is_file():
        raise FileNotFoundError(f"Missing supervised checkpoint: {sup_path}")
    if ("joint_frozen" in resolved_conditions or "online_ttt" in resolved_conditions) and not jnt_path.is_file():
        raise FileNotFoundError(f"Missing joint checkpoint: {jnt_path}")

    schema_map = load_schema_map(schema_map_path)
    dev = resolve_device(resolved_device)
    ver = get_code_version_strict()

    if max_adapt_images is not None and max_adapt_images <= 0:
        raise ValueError(f"max_adapt_images must be positive, got {max_adapt_images}")

    adapt_refs, held_out_refs, split_meta = load_exp_b_split_pair(
        split_file=resolved_split_file,
        held_out_file=resolved_held_out_file,
        adaptation_key=adaptation_key,
        held_out_key=held_out_key,
        train_val_membership=resolved_train_val,
        prior_test_membership=resolved_prior_test,
    )
    if max_adapt_images is not None:
        adapt_refs = adapt_refs[:max_adapt_images]
    split_meta["adaptation_count_effective"] = len(adapt_refs)

    shared_ds: Dict[str, Any] = dict(datasets_by_split) if datasets_by_split is not None else {}

    ho_clean_batches: Optional[List[Tuple[torch.Tensor, torch.Tensor, Tuple[str, ...]]]] = None
    ho_clean_ids: Optional[Tuple[str, ...]] = None
    ho_noisy_batches: Optional[List[Tuple[torch.Tensor, torch.Tensor, Tuple[str, ...]]]] = None
    ho_noisy_ids: Optional[Tuple[str, ...]] = None

    if "clean" in resolved_views:
        ho_clean_ds = CIFAR100SplitSubsetDataset(
            held_out_refs,
            corruption="clean",
            noise_sigma=resolved_noise_sigma,
            noise_seed=resolved_noise_seed,
            datasets_by_split=shared_ds if shared_ds else None,
        )
        shared_ds.update(ho_clean_ds.datasets_by_split)
        ho_clean_batches, ho_clean_ids = prepare_held_out_batches(
            ho_clean_ds, batch_size=resolved_eval_batch_size
        )

    if "noisy" in resolved_views:
        ho_noisy_ds = CIFAR100SplitSubsetDataset(
            held_out_refs,
            corruption="gaussian_noise",
            noise_sigma=resolved_noise_sigma,
            noise_seed=resolved_noise_seed,
            datasets_by_split=shared_ds if shared_ds else None,
        )
        shared_ds.update(ho_noisy_ds.datasets_by_split)
        ho_noisy_batches, ho_noisy_ids = prepare_held_out_batches(
            ho_noisy_ds, batch_size=resolved_eval_batch_size
        )

    adapt_loaders: Dict[str, DataLoader] = {}
    if "online_ttt" in resolved_conditions:
        for stream_cond in resolved_streams:
            stream_corr = "clean" if stream_cond == "clean" else "gaussian_noise"
            adapt_ds = CIFAR100SplitSubsetDataset(
                adapt_refs,
                corruption=stream_corr,
                noise_sigma=resolved_noise_sigma,
                noise_seed=resolved_noise_seed,
                datasets_by_split=shared_ds if shared_ds else None,
            )
            shared_ds.update(adapt_ds.datasets_by_split)
            adapt_loaders[stream_cond] = DataLoader(
                adapt_ds,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )

    split_desc = (
        "<in-memory>"
        if isinstance(resolved_split_file, dict)
        else str(resolved_split_file)
    )
    if resolved_held_out_file is None:
        split_loc_str = split_desc
    else:
        ho_desc = (
            "<in-memory>"
            if isinstance(resolved_held_out_file, dict)
            else str(resolved_held_out_file)
        )
        split_loc_str = f"{split_desc};{ho_desc}"

    all_rows: List[Dict[str, Any]] = []
    run_audits: List[Dict[str, Any]] = []
    resolved_num_outputs: Optional[int] = None

    run_specs: List[Tuple[str, str, Path, Optional[DataLoader]]] = []
    for cond_name in resolved_conditions:
        if cond_name == "supervised_frozen":
            run_specs.append(("supervised_frozen", "none", sup_path, None))
        elif cond_name == "joint_frozen":
            run_specs.append(("joint_frozen", "none", jnt_path, None))
        elif cond_name == "online_ttt":
            for stream_cond in resolved_streams:
                run_specs.append(("online_ttt", stream_cond, jnt_path, adapt_loaders[stream_cond]))

    for cond_name, stream_cond, ckpt_path, adapt_loader in run_specs:
        # Load a fresh source checkpoint for every run (frozen or adaptive stream)
        fresh_model, ckpt_meta = load_model(
            ckpt_path,
            device=dev,
            model_config=cfg,
        )
        model_out_dim = int(fresh_model.classifier[-1].out_features)
        if resolved_num_outputs is None:
            resolved_num_outputs = model_out_dim
        elif model_out_dim != resolved_num_outputs:
            raise ValueError(
                f"Checkpoint '{ckpt_path}' has num_outputs={model_out_dim}, "
                f"expected {resolved_num_outputs}"
            )
        _validate_checkpoint_metadata(
            ckpt_meta,
            ckpt_path=ckpt_path,
            condition=cond_name,
            expected_seed=resolved_seed,
        )
        rows, audit = run_held_out_stream(
            model=fresh_model,
            adaptation_loader=adapt_loader,
            held_out_clean_batches=ho_clean_batches,
            held_out_clean_ids=ho_clean_ids,
            held_out_noisy_batches=ho_noisy_batches,
            held_out_noisy_ids=ho_noisy_ids,
            held_out_views=resolved_views,
            condition=cond_name,
            adaptation_stream_condition=stream_cond,
            eval_interval=resolved_eval_interval,
            ttt_lr=resolved_ttt_lr,
            ttt_steps=resolved_ttt_steps,
            ttt_momentum=resolved_ttt_momentum,
            lambda_rot=resolved_lambda_rot,
            seed=resolved_seed,
            source_checkpoint=str(ckpt_path),
            config_path=config_path_str,
            split_locations=split_loc_str,
            code_version=ver,
            device=dev,
            verify_invariants=verify_invariants,
            max_adapt_images=max_adapt_images,
            expected_num_outputs=model_out_dim,
        )
        all_rows.extend(rows)
        run_audits.append(audit)

    out_csv_path = save_held_out_csv(
        all_rows,
        resolved_output_csv,
        schema_map=schema_map,
        overwrite=overwrite,
    )
    checks_report = {
        "experiment": "B",
        "seed": resolved_seed,
        "device": str(dev.type),
        "code_version": ver,
        "config_path": config_path_str,
        "split_locations": split_loc_str,
        "output_csv": str(out_csv_path),
        "output_checks": str(checks_path),
        "timing_semantics": {
            "adaptation_time_seconds": (
                "Cumulative adaptation wall time since stream start for this condition; "
                "identical across clean/noisy held-out rows at the same checkpoint (do not sum across rows)."
            ),
            "adaptation_interval_time_seconds": (
                "Adaptation wall time since the previous checkpoint for this stream."
            ),
            "held_out_eval_time_seconds": (
                "Wall time for this individual held-out evaluation pass (per view, per checkpoint)."
            ),
            "cumulative_held_out_eval_time_seconds": (
                "Cumulative held-out evaluation wall time across checkpoints for this held_out_condition view."
            ),
        },
        "settings": {
            "conditions": list(resolved_conditions),
            "adaptation_streams": list(resolved_streams),
            "held_out_views": list(resolved_views),
            "ttt_optimizer": ttt_optimizer,
            "eval_interval": resolved_eval_interval,
            "eval_batch_size": resolved_eval_batch_size,
            "ttt_lr": resolved_ttt_lr,
            "ttt_steps": resolved_ttt_steps,
            "ttt_momentum": resolved_ttt_momentum,
            "lambda_rot": resolved_lambda_rot,
            "noise_sigma": resolved_noise_sigma,
            "noise_seed": resolved_noise_seed,
            "max_adapt_images": max_adapt_images,
        },
        "split_checks": split_meta,
        "runs": run_audits,
        "total_rows_written": len(all_rows),
        "all_checks_passed": all(r["all_invariants_passed"] for r in run_audits),
    }

    checks_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_checks = checks_path.with_name(f".{checks_path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp_checks, "w", encoding="utf-8") as handle:
            json.dump(checks_report, handle, indent=2)
        os.replace(tmp_checks, checks_path)
    finally:
        if tmp_checks.exists():
            tmp_checks.unlink()

    return all_rows, checks_report


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Experiment B: Periodic Held-Out Evaluation for Online TTT"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_HELD_OUT_CONFIG,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="Path to Khoi's split JSON (either combined adaptation+held-out or adaptation-only)",
    )
    parser.add_argument(
        "--held-out-file",
        type=str,
        default=None,
        help="Optional separate JSON file for fixed held-out evaluation IDs",
    )
    parser.add_argument(
        "--adaptation-key",
        type=str,
        default=None,
        help="Optional JSON key for adaptation IDs",
    )
    parser.add_argument(
        "--held-out-key",
        type=str,
        default=None,
        help="Optional JSON key for held-out IDs",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV path (defaults to config or results/exp_b_held_out_pilot.csv)",
    )
    parser.add_argument(
        "--schema-map",
        type=str,
        default=None,
        help="Optional JSON file mapping canonical column names to Alka's agreed column names",
    )
    parser.add_argument(
        "--supervised-ckpt",
        type=str,
        default=None,
        help="Override path to seed supervised checkpoint",
    )
    parser.add_argument(
        "--joint-ckpt",
        type=str,
        default=None,
        help="Override path to seed joint checkpoint",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-adapt-images", type=int, default=None)
    parser.add_argument("--ttt-lr", type=float, default=None)
    parser.add_argument("--ttt-steps", type=int, default=None)
    parser.add_argument("--ttt-momentum", type=float, default=None)
    parser.add_argument("--noise-sigma", type=float, default=None)
    parser.add_argument("--noise-seed", type=int, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "mps", "cuda", "cpu"],
    )
    parser.add_argument(
        "--skip-invariant-checks",
        action="store_true",
        help="Disable per-checkpoint model/optimizer/RNG snapshot assertions",
    )
    parser.add_argument(
        "--check-splits-only",
        action="store_true",
        help="Validate split disjointness and prior-data overlap without running model evaluation",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output CSV (legacy outputs remain protected)",
    )

    args = parser.parse_args(argv)

    cfg = _load_yaml_mapping(args.config)
    paths_cfg = _cfg_section(cfg, "paths")

    if args.split_file is not None and not args.split_file.strip():
        raise ValueError("--split-file must be a non-empty path")
    if args.held_out_file is not None and not args.held_out_file.strip():
        raise ValueError("--held-out-file must be a non-empty path")
    resolved_cli_split = (
        args.split_file
        if args.split_file is not None
        else _cfg_path_or(paths_cfg, "split_file", None)
    )
    if resolved_cli_split is None:
        parser.error("the following arguments are required: --split-file")
    resolved_cli_held_out = (
        args.held_out_file
        if args.held_out_file is not None
        else _cfg_path_or(paths_cfg, "held_out_file", None)
    )

    if args.check_splits_only:
        _, _, split_meta = load_exp_b_split_pair(
            resolved_cli_split,
            resolved_cli_held_out,
            adaptation_key=args.adaptation_key,
            held_out_key=args.held_out_key,
            train_val_membership=_cfg_path_or(
                paths_cfg, "shared_c20_membership", SHARED_C20_MEMBERSHIP_PATH
            ),
            prior_test_membership=_cfg_path_or(
                paths_cfg, "prior_exp_b_test_membership", EXP_B_TEST_MEMBERSHIP_PATH
            ),
        )
        print(json.dumps(split_meta, indent=2))
        return

    rows, report = run_held_out_experiment(
        split_file=resolved_cli_split,
        held_out_file=resolved_cli_held_out,
        adaptation_key=args.adaptation_key,
        held_out_key=args.held_out_key,
        config_path=args.config,
        output_csv=args.output_csv,
        supervised_ckpt=args.supervised_ckpt,
        joint_ckpt=args.joint_ckpt,
        seed=args.seed,
        eval_interval=args.eval_interval,
        eval_batch_size=args.eval_batch_size,
        ttt_lr=args.ttt_lr,
        ttt_steps=args.ttt_steps,
        ttt_momentum=args.ttt_momentum,
        noise_sigma=args.noise_sigma,
        noise_seed=args.noise_seed,
        device=args.device,
        max_adapt_images=args.max_adapt_images,
        verify_invariants=not args.skip_invariant_checks,
        schema_map_path=args.schema_map,
        overwrite=args.overwrite,
    )

    print(f"[Held-Out Runner] Saved {len(rows)} rows to {report['output_csv']}")
    if report["split_checks"].get("held_out_examined_in_prior_exp_b"):
        overlap_n = report["split_checks"]["held_out_prior_overlap_count"]
        print(
            f"[Notice] {overlap_n} held-out image(s) overlap data/exp_b_test_membership.json "
            "(already examined in earlier Experiment B runs; do not describe them as completely new data)."
        )
    for r in rows:
        print(
            f"  {r['condition']:17s} | stream={r['adaptation_stream_condition']:5s} | "
            f"eval={r['held_out_condition']:5s} | n={r['adaptation_images_processed']:4d} | "
            f"acc={r['held_out_accuracy']:.4f} ({r['correct_prediction_count']}/{r['held_out_image_count']}) | "
            f"adapt_t={r['adaptation_time_seconds']:.3f}s | eval_t={r['held_out_eval_time_seconds']:.3f}s"
        )


if __name__ == "__main__":
    main()
