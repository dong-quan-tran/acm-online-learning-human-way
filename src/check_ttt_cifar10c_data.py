"""Check the official CIFAR-10 / CIFAR-10-C data alignment without a model."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from src.ttt_evaluation import cifar10c_image, save_manifest, validate_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("results/ttt_cifar10c_data_pilot.json"))
    args = parser.parse_args()
    corruption_dir = args.root / "CIFAR-10-C"
    required = [args.root / "cifar-10-batches-py" / "test_batch",
                corruption_dir / "labels.npy", corruption_dir / "gaussian_noise.npy"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("CIFAR-10-C pilot inputs are not ready. Wait for download and extraction.\n"
                         "Missing:\n  " + "\n  ".join(missing))
    try:
        clean = CIFAR10(args.root, train=False, download=False, transform=ToTensor())
    except RuntimeError as error:
        raise SystemExit(f"CIFAR-10 could not be loaded from {args.root}: {error}") from None
    labels = np.load(corruption_dir / "labels.npy", allow_pickle=False)
    severity, seed = 1, 42
    if labels.shape not in {(10000,), (50000,)}:
        raise ValueError(f"Unexpected CIFAR-10-C label shape: {labels.shape}")
    offset = (severity - 1) * 10000
    block = labels if len(labels) == 10000 else labels[offset:offset + 10000]
    if not np.array_equal(block, clean.targets):
        raise ValueError("Official clean and corrupted class labels differ in row order")

    rng = random.Random(seed)
    development, evaluation = [], []
    for label in range(10):
        dev, test = rng.sample([i for i, y in enumerate(clean.targets) if y == label], 2)
        development.append(f"test:{dev}")
        evaluation.append(f"test:{test}")
    rng.shuffle(evaluation)
    validate_split(development, evaluation)
    for source_id in evaluation:
        row = int(source_id.split(":")[1])
        image, _ = clean[row]
        noisy = cifar10c_image(corruption_dir / "gaussian_noise.npy", row, severity)
        if image.shape != (3, 32, 32) or noisy.shape != image.shape or torch.equal(image, noisy):
            raise ValueError(f"Invalid clean/corrupted pair for {source_id}")

    record = dict(dataset="Official CIFAR-10 and CIFAR-10-C; data-only pilot",
                  development_ids=development, evaluation_ids=evaluation, seed=seed,
                  preprocessing={"shape": [3, 32, 32], "range": [0, 1], "normalization": "none"},
                  corruption={"name": "gaussian_noise", "severity": severity,
                              "source": "https://zenodo.org/records/2535967"})
    if args.manifest.exists():
        if json.loads(args.manifest.read_text(encoding="utf-8")) != record:
            raise ValueError("Existing manifest differs from regenerated pilot")
    else:
        save_manifest(args.manifest, **record)
    print("PASS: all 10000 clean and severity-1 labels agree in official row order")
    print("PASS: 10 paired RGB evaluation images; zero overlap with 10 development IDs")
    print(f"Verified manifest: {args.manifest}")


if __name__ == "__main__":
    main()
