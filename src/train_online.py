"""Reusable one-pass training/evaluation and a configured MNIST entry point."""
import argparse
import csv
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml

from src.mnist_data import get_mnist_loader
from src.models import SmallResNet


def classification_metrics(confusion):
    matrix = confusion.to(torch.float64)
    tp = matrix.diag()
    predicted = matrix.sum(0)
    actual = matrix.sum(1)
    precision = torch.where(predicted > 0, tp / predicted, 0)
    recall = torch.where(actual > 0, tp / actual, 0)
    f1 = torch.where(precision + recall > 0,
                     2 * precision * recall / (precision + recall), 0)
    return {"accuracy": (tp.sum() / matrix.sum()).item(),
            "precision": precision.mean().item(), "recall": recall.mean().item(),
            "f1": f1.mean().item()}


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """Evaluate without gradients or BatchNorm updates; restore training mode."""
    was_training = model.training
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    try:
        for images, labels, _ in loader:
            predictions = model(images.to(device)).argmax(1).cpu()
            confusion += torch.bincount(labels.cpu() * num_classes + predictions,
                                       minlength=num_classes ** 2).reshape(num_classes, num_classes)
    finally:
        model.train(was_training)
    if confusion.sum() == 0:
        raise ValueError("evaluation loader is empty")
    return classification_metrics(confusion)


def train_one_pass(model, loader, optimizer, device, checkpoint, eval_every_examples):
    """Consume the loader once, one update per batch, retaining no images."""
    if eval_every_examples <= 0:
        raise ValueError("eval_every_examples must be positive")
    examples, updates, training_seconds = 0, 0, 0.0
    next_evaluation = eval_every_examples
    checkpoint(0, 0, 0.0)
    last_evaluated = 0
    for images, labels, _ in loader:
        model.train()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(images.to(device)), labels.to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        optimizer.step()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        training_seconds += time.perf_counter() - started
        examples += len(labels)
        updates += 1
        if examples >= next_evaluation:
            checkpoint(examples, updates, training_seconds)
            last_evaluated = examples
            next_evaluation = (examples // eval_every_examples + 1) * eval_every_examples
        if updates % 100 == 0:
            print(f"Progress: {examples} examples, {updates} updates", flush=True)
    if examples == 0:
        raise ValueError("training loader is empty")
    if examples != last_evaluated:
        checkpoint(examples, updates, training_seconds)
    return examples, updates, training_seconds


def run(config_path, output, download=False):
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if (config["dataset"] != "mnist" or config["model"] != "SmallResNet"
            or config["passes"] != 1 or config["replay"] or config["batch_size"] != 32
            or config["preprocessing"] != "ToTensor" or config["optimizer"] != "adam"
            or config["in_channels"] != 1 or config["num_classes"] != 10):
        raise ValueError("unsupported configuration for the MNIST one-pass experiment")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    random.seed(config["model_seed"])
    np.random.seed(config["model_seed"])
    torch.manual_seed(config["model_seed"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    # CPU makes the timing scope and deterministic execution explicit.
    device = torch.device("cpu")
    kwargs = {key: config[key] for key in ("split_seed", "stream_seed", "validation_size")}
    train = get_mnist_loader(batch_size=config["batch_size"], download=download, **kwargs)
    validation = get_mnist_loader("validation", batch_size=config["batch_size"],
                                   download=download, **kwargs)
    membership = {"source_split": "train", "train_stream_rows": train.dataset.indices,
                  "validation_rows": validation.dataset.indices}
    (output / "membership.json").write_text(json.dumps(membership), encoding="utf-8")
    metadata = {"config": config, "command": [sys.executable, *sys.argv],
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "device": str(device), "platform": platform.platform(),
                "processor": platform.processor(), "python": sys.version,
                "packages": {name: version(name) for name in
                             ("torch", "torchvision", "numpy", "PyYAML", "matplotlib")},
                "metrics": "macro precision, recall, F1 across all ten classes; zero division = 0",
                "training_seconds": "forward/backward/optimizer time, excluding evaluation and loading",
                "comparison_status": "provisional: shared CIFAR online trainer/settings unpublished"}
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    model = SmallResNet(num_classes=config["num_classes"], in_channels=config["in_channels"],
                        dropout=config["dropout"], mean=config["mean"], std=config["std"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    fields = ["mode", "run_id", "model_seed", "stream_seed", "num_classes",
              "examples_processed", "updates", "passes", "batch_size", "learning_rate",
              "optimizer", "eval_split", "accuracy", "training_seconds", "precision", "recall", "f1"]
    wall_start = time.perf_counter()
    with (output / "results.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        def checkpoint(examples, updates, seconds):
            metrics = evaluate(model, validation, device, config["num_classes"])
            row = {"mode": "online", "run_id": output.name,
                   "model_seed": config["model_seed"], "stream_seed": config["stream_seed"],
                   "num_classes": config["num_classes"], "examples_processed": examples,
                   "updates": updates, "passes": 1, "batch_size": config["batch_size"],
                   "learning_rate": config["learning_rate"], "optimizer": config["optimizer"],
                   "eval_split": "validation", "training_seconds": seconds, **metrics}
            writer.writerow(row)
            handle.flush()
            print(f"Validation: examples={examples} updates={updates} "
                  f"accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}", flush=True)

        examples, updates, seconds = train_one_pass(
            model, train, optimizer, device, checkpoint, config["eval_every_examples"])
    metadata.update(examples_processed=examples, updates=updates, training_seconds=seconds,
                    total_wall_seconds=time.perf_counter() - wall_start)
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), output / "model.pt")
    # Reuse the repository plotter with isolated input/output locations.
    from src import plot_results
    plot_results.RESULTS_PATH = output / "results.csv"
    plot_results.FIGURES_PATH = output
    plot_results.plot_learning_curve()
    print(f"Completed: {examples} examples, {updates} updates; saved to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mnist.yaml")
    parser.add_argument("--output", required=True, help="New run directory; existing paths are refused")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output, args.download)
