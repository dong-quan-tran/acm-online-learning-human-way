
import argparse
import csv
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.naive_bayes import GaussianNB

from data import get_online_loader


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "gaussiannb_results.csv"

FIELDS = [
    "mode", "run_id", "model_seed", "stream_seed",
    "num_classes", "examples_processed", "updates",
    "passes", "batch_size", "learning_rate", "optimizer",
    "eval_split", "accuracy", "training_seconds",
]


def flatten_images(images):
    """Convert CIFAR images into flattened, scaled pixel features."""
    return images.reshape(images.shape[0], -1).numpy()


def evaluate(model, loader):
    correct = 0
    total = 0

    for images, labels, _ in loader:
        features = flatten_images(images)
        predictions = model.predict(features)

        if not np.all(np.isfinite(predictions)):
            raise ValueError("Model produced invalid predictions")

        correct += np.sum(predictions == labels.numpy())
        total += len(labels)

    return float(correct / total)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader = get_online_loader(
        split="train",
        batch_size=args.batch_size,
        seed=args.seed,
        membership=args.membership,
    )
    validation_loader = get_online_loader(
        split="validation",
        batch_size=args.batch_size,
        seed=args.seed,
        membership=args.membership,
    )

    model = GaussianNB()
    classes = np.arange(20)
    checkpoints = set(args.checkpoints)

    processed = 0
    updates = 0
    rows = []
    start = time.perf_counter()

    # No zero-example accuracy: GaussianNB needs its first update.
    for images, labels, _ in train_loader:
        features = flatten_images(images)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.partial_fit(
                features,
                labels.numpy(),
                classes=classes if updates == 0 else None,
            )

        for warning in caught:
            print(f"Training warning: {warning.message}")

        processed += len(labels)
        updates += 1

        if processed in checkpoints or processed == len(train_loader.dataset):
            accuracy = evaluate(model, validation_loader)
            elapsed = time.perf_counter() - start

            rows.append({
                "mode": "online",
                "run_id": args.run_id,
                "model_seed": args.seed,
                "stream_seed": args.seed,
                "num_classes": len(classes),
                "examples_processed": processed,
                "updates": updates,
                "passes": processed / len(train_loader.dataset),
                "batch_size": args.batch_size,
                "learning_rate": "",
                "optimizer": "GaussianNB partial_fit",
                "eval_split": "validation",
                "accuracy": accuracy,
                "training_seconds": elapsed,
            })
            print(f"Examples: {processed}, accuracy: {accuracy:.4f}")

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results saved to {CSV_PATH}")
    print("Device: CPU")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="gaussiannb_seed42")
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[1000, 2000, 4000, 6000, 8000, 9000],
    )
    train(parser.parse_args())
