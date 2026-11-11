"""Shared, label-free adaptation and evaluation for a saved image stream."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def save_manifest(path, *, dataset, development_ids, evaluation_ids, seed, preprocessing, corruption):
    """IDs refer to source images: variants of the same image share an ID."""
    development_ids, evaluation_ids = list(development_ids), list(evaluation_ids)
    validate_split(development_ids, evaluation_ids)
    record = dict(dataset=dataset, development_ids=development_ids,
                  evaluation_ids=evaluation_ids, seed=seed,
                  preprocessing=preprocessing, corruption=corruption)
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return record


def validate_split(development_ids, evaluation_ids):
    dev, test = set(development_ids), set(evaluation_ids)
    if not dev or not test:
        raise ValueError("Development and evaluation must both be nonempty")
    if len(dev) != len(development_ids) or len(test) != len(evaluation_ids):
        raise ValueError("Duplicate source image IDs in a split")
    if dev & test:
        raise ValueError("Development and evaluation source images overlap")


def cifar10c_image(corruption_file, row, severity):
    """Read one CIFAR-10-C image aligned to its official CIFAR-10 test row."""
    if not 1 <= severity <= 5 or not 0 <= row < 10000:
        raise ValueError("Expected severity 1-5 and official test row 0-9999")
    images = np.load(corruption_file, mmap_mode="r", allow_pickle=False)
    if images.shape != (50000, 32, 32, 3) or images.dtype != np.uint8:
        raise ValueError("Expected a CIFAR-10-C uint8 corruption array")
    index = (severity - 1) * 10000 + row
    return torch.from_numpy(images[index].copy()).permute(2, 0, 1).float() / 255


def gaussian_noise(image, source_id, *, seed, std):
    """Apply deterministic noise to a float CPU image in [0, 1] before normalization."""
    if std < 0:
        raise ValueError("Noise strength must be nonnegative")
    digest = hashlib.sha256(f"{seed}:{source_id}".encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
    noise = torch.randn(image.shape, generator=generator, dtype=image.dtype)
    return (image + std * noise).clamp(0, 1)


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def evaluate_stream(model, optimizer, *, images, evaluation_ids, predict, condition,
                    adapt=None):
    """Return (source ID, predicted class) rows; no class labels enter this API.

    images maps source IDs to already prepared image tensors. Callbacks take
    predict(model, image) -> class ID and adapt(model, optimizer, image).
    Supply a pristine checkpoint/optimizer pair; each call copies the pair
    together so optimizer parameter references follow the copied model.
    """
    if condition not in {"frozen", "standard", "online"}:
        raise ValueError("Unknown evaluation condition")
    if condition != "frozen" and adapt is None:
        raise ValueError("Adaptive conditions require an adaptation callback")
    if not evaluation_ids or len(set(evaluation_ids)) != len(evaluation_ids):
        raise ValueError("Evaluation order must be nonempty with unique IDs")
    model, optimizer = deepcopy((model, optimizer))
    model.eval()
    initial_model = deepcopy(model.state_dict())
    initial_optimizer = deepcopy(optimizer.state_dict())
    rows = []
    for source_id in evaluation_ids:
        image = images[source_id]
        if condition != "frozen":
            adapt(model, optimizer, image.clone())
        model.eval()
        # Prediction itself must not update parameters, buffers or optimizer.
        before_model = deepcopy(model.state_dict())
        before_optimizer = deepcopy(optimizer.state_dict())
        with torch.no_grad():
            prediction = int(predict(model, image.clone()))
        if not (_equal(before_model, model.state_dict())
                and _equal(before_optimizer, optimizer.state_dict())):
            raise ValueError("Prediction changed model or optimizer state")
        rows.append((source_id, prediction))
        if condition == "standard":
            model.load_state_dict(initial_model)
            # Optimizer loading can alias state tensors; copy the reset snapshot.
            optimizer.load_state_dict(deepcopy(initial_optimizer))
            model.zero_grad(set_to_none=True)
    return rows


def score_predictions(rows, labels, evaluation_ids):
    """Use class labels only after prediction; require exactly one row per ID."""
    rows = list(rows)
    expected = list(evaluation_ids)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("Expected evaluation IDs must be nonempty and unique")
    if Counter(source_id for source_id, _ in rows) != Counter(expected):
        raise ValueError("Each evaluation image must have exactly one prediction")
    return sum(prediction == labels[source_id] for source_id, prediction in rows) / len(expected)
