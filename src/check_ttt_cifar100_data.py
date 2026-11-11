"""Reproduce the small, data-only CIFAR-100 classes 0-19 TTT pilot."""
import argparse
import json
import random
from pathlib import Path

import torch
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor

from src.ttt_evaluation import gaussian_noise, save_manifest, validate_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("results/ttt_cifar100_data_pilot.json"))
    args = parser.parse_args()
    seed, std = 42, 0.1
    dataset = CIFAR100(args.root, train=False, download=False, transform=ToTensor())
    rng = random.Random(seed)
    development, evaluation = [], []
    for label in range(20):
        candidates = [i for i, target in enumerate(dataset.targets) if target == label]
        dev, test = rng.sample(candidates, 2)
        development.append(f"test:{dev}")
        evaluation.append(f"test:{test}")
    rng.shuffle(evaluation)
    validate_split(development, evaluation)

    clean, noisy = {}, {}
    for source_id in evaluation:
        image, _ = dataset[int(source_id.split(":")[1])]
        if image.shape != (3, 32, 32) or not torch.all((image >= 0) & (image <= 1)):
            raise RuntimeError(f"Unexpected clean image shape or range: {source_id}")
        clean[source_id] = image
        noisy[source_id] = gaussian_noise(image, source_id, seed=seed, std=std)
        if torch.equal(image, noisy[source_id]) or not torch.all(
            (noisy[source_id] >= 0) & (noisy[source_id] <= 1)
        ):
            raise RuntimeError(f"Noise did not produce a valid changed image: {source_id}")
    if list(clean) != list(noisy) or list(clean) != evaluation:
        raise RuntimeError("Clean and noisy streams are not aligned")
    for source_id in reversed(evaluation):
        replay = gaussian_noise(clean[source_id], source_id, seed=seed, std=std)
        if not torch.equal(replay, noisy[source_id]):
            raise RuntimeError(f"Noise depends on processing order: {source_id}")

    record = dict(dataset="torchvision CIFAR-100 official test, fine classes 0-19; data-only pilot",
                  development_ids=development, evaluation_ids=evaluation, seed=seed,
                  preprocessing={"transform": "ToTensor", "shape": [3, 32, 32],
                                 "range": [0, 1], "normalization": "none"},
                  corruption={"kind": "Gaussian noise", "std": std, "seed": seed,
                              "clip": [0, 1], "seed_scope": "source image ID"})
    if args.manifest.exists():
        if json.loads(args.manifest.read_text(encoding="utf-8")) != record:
            raise RuntimeError("Existing manifest differs from regenerated pilot")
        action = "Verified"
    else:
        save_manifest(args.manifest, **record)
        action = "Saved"
    print(f"PASS: {len(development)} development and {len(evaluation)} evaluation images, "
          "one per class in each split, zero overlapping source IDs")
    print(f"PASS: {len(clean)} aligned clean/noisy pairs; noise reproduced exactly in reverse order")
    print(f"{action} manifest: {args.manifest}")


if __name__ == "__main__":
    main()
