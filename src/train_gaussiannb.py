
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
    """Flatten CIFAR images, preserving the loader's pixel scaling."""
    return images.reshape(images.shape[0], -1).cpu().numpy()


def evaluate(model, loader):
    """Calculate validation accuracy using the current model."""
    correct = 0
    total = 0

    for images, labels, _ in loader:
        features = flatten_images(images)
        predictions = model.predict(features)

        if not np.all(np.isfinite(predictions)):
            raise ValueError("Model produced invalid predictions")

        correct += int(np.sum(predictions == labels.numpy()))
        total += len(labels)

    if total == 0:
        raise ValueError("Validation loader is empty")

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
    training_seconds = 0.0

    # GaussianNB cannot predict before its first partial_fit update.
    for images, labels, _ in train_loader:
        features = flatten_images(images)

        # Measure only the model update, not validation.
        batch_start = time.perf_counter()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            model.partial_fit(
                features,
                labels.numpy(),
                classes=classes if updates == 0 else None,
            )

        training_seconds += time.perf_counter() - batch_start

        for warning in caught:
            print(f"Training warning: {warning.message}")

        previous = processed
        processed += len(labels)
        updates += 1

        # Evaluate when a batch crosses a checkpoint.
        reached_checkpoint = any(
            previous < cp <= processed
            for cp in checkpoints
        )

        completed_pass = processed == len(train_loader.dataset)

        if reached_checkpoint or completed_pass:
            accuracy = evaluate(model, validation_loader)

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
                "training_seconds": training_seconds,
            })

            print(
                f"Examples: {processed}, "
                f"accuracy: {accuracy:.4f}, "
                f"training time: {training_seconds:.3f}s"
            )

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results saved to {CSV_PATH}")
    print(f"Total training time: {training_seconds:.3f}s")
    print("Device: CPU")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Online Gaussian Naive Bayes on CIFAR-100"
    )

    parser.add_argument("--membership", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="gaussiannb_seed42")

    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[
            1024, 2016, 3008, 4000, 5024,
            6016, 7008, 8000, 9000,
        ],
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    train(args)
