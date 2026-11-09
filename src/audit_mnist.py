"""Audit a saved MNIST checkpoint without training or official test evaluation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from src.data import DATA_PATH
from src.mnist_data import MNISTStream, get_mnist_transform
from src.models import SmallResNet
from src.train_online import classification_metrics


def audit(run_dir, output):
    run_dir, output = Path(run_dir), Path(output)
    metadata = json.loads((run_dir / "run.json").read_text())
    config = metadata["config"]
    membership = json.loads((run_dir / "membership.json").read_text())
    training, validation = membership["train_stream_rows"], membership["validation_rows"]
    assert membership["source_split"] == "train"
    expected_validation = config["validation_size"]
    expected_training = config.get("train_size", 60000 - expected_validation)
    assert len(training) == len(set(training)) == expected_training
    assert len(validation) == len(set(validation)) == expected_validation
    assert not set(training) & set(validation)
    assert set(training) | set(validation) <= set(range(60000))
    assert len(set(training) | set(validation)) == expected_training + expected_validation
    # Check configured loader agrees, but evaluate directly from the saved rows.
    for split, expected in (("train", training), ("validation", validation)):
        dataset = MNISTStream(split, split_seed=config["split_seed"],
                              stream_seed=config["stream_seed"],
                              validation_size=config["validation_size"],
                              train_size=config.get("train_size"), download=False,
                              profile=config.get("profile", "native"),
                              membership_strategy=config.get("membership_strategy", "random"))
        assert dataset.indices == expected, f"{split} loader differs from saved membership"
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    source = MNIST(DATA_PATH, train=True, download=False, transform=get_mnist_transform(config.get("profile", "native"), training=False))
    assert set(source.targets[training].tolist()) == set(range(10))
    assert set(source.targets[validation].tolist()) == set(range(10))
    training_class_counts = torch.bincount(source.targets[training], minlength=10).tolist()
    validation_class_counts = torch.bincount(source.targets[validation], minlength=10).tolist()
    if config.get("membership_strategy") == "nguyen_stratified":
        assert training_class_counts == [expected_training // 10] * 10
        assert validation_class_counts == [expected_validation // 10] * 10
    loader = DataLoader(Subset(source, validation), batch_size=config.get("eval_batch_size", 32), shuffle=False, drop_last=False)
    outputs = config.get("num_outputs", config["num_classes"])
    model = SmallResNet(num_classes=outputs, in_channels=config["in_channels"],
                        dropout=config["dropout"], mean=config["mean"], std=config["std"])
    checkpoint = run_dir / "model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    confusion = torch.zeros(outputs, outputs, dtype=torch.int64)
    predictions, errors, seen, correct = [], [], 0, 0
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images)
            assert logits.shape == (len(labels), outputs) and torch.isfinite(logits).all()
            predicted = logits.argmax(1)
            confusion += torch.bincount(labels * outputs + predicted, minlength=outputs ** 2).reshape(outputs, outputs)
            correct += int((predicted == labels).sum())
            for i, (label, pred) in enumerate(zip(labels.tolist(), predicted.tolist())):
                row = validation[seen+i]
                predictions.append({"sample_id": f"train:{row}", "label": label, "prediction": pred})
                if label != pred and len(errors) < 20:
                    display_image = images[i].squeeze().clone()
                    if display_image.ndim == 3:
                        display_image = display_image.permute(1, 2, 0)
                    errors.append((display_image, label, pred, row))
            seen += len(labels)
            if seen % 1024 == 0:
                print(f"Audited {seen}/{expected_validation} validation images", flush=True)
    assert seen == expected_validation and all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    metrics = classification_metrics(confusion, label_classes=range(10))
    assert metrics["accuracy"] == correct / seen
    # Separate per-class calculation as a metric cross-check.
    ps, rs, fs = [], [], []
    for digit in range(10):
        tp = int(confusion[digit, digit])
        p = tp / int(confusion[:,digit].sum()) if confusion[:,digit].sum() else 0
        r = tp / int(confusion[digit,:].sum()) if confusion[digit,:].sum() else 0
        ps.append(p); rs.append(r); fs.append(2*p*r/(p+r) if p+r else 0)
    for name, value in (("precision", sum(ps)/10), ("recall", sum(rs)/10), ("f1", sum(fs)/10)):
        assert abs(metrics[name]-value) < 1e-12
    with (run_dir / "results.csv").open(newline="") as handle:
        final = list(csv.DictReader(handle))[-1]
    assert int(final["examples_processed"]) == expected_training
    assert int(final["updates"]) == (expected_training + 31) // 32
    for key, value in metrics.items():
        assert abs(value-float(final[key])) < 1e-12, f"Saved {key} does not reproduce"
    output.mkdir(parents=True, exist_ok=False)
    report = {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "validation_examples": seen, "correct": correct, "mistakes": seen-correct,
              "train_validation_overlap": 0, "loader_membership_matches": True,
              "model_state_unchanged": True, "saved_metrics_match_tolerance": 1e-12,
              "official_test_set_used": False, "num_outputs": outputs,
              "macro_label_classes": list(range(10)),
              "training_class_counts": training_class_counts,
              "validation_class_counts": validation_class_counts,
              "predictions_outside_digit_classes": int(confusion[:,10:].sum()),
              "metrics": metrics,
              "confusion_matrix": confusion.tolist()}
    (output / "audit.json").write_text(json.dumps(report, indent=2)+"\n")
    with (output / "validation_predictions.csv").open("w",newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id","label","prediction"])
        writer.writeheader(); writer.writerows(predictions)
    fig, ax = plt.subplots(figsize=(8,7))
    display_confusion = confusion[:10, :10]
    if outputs > 10:
        display_confusion = torch.cat((display_confusion, confusion[:10,10:].sum(1,keepdim=True)),1)
    ax.imshow(display_confusion.numpy(),cmap="Blues")
    for i in range(10):
        for j in range(display_confusion.shape[1]):
            ax.text(j,i,str(int(display_confusion[i,j])),ha="center",va="center",
                    color="white" if confusion[i,j]>350 else "black",fontsize=8)
    ax.set(xticks=range(display_confusion.shape[1]),yticks=range(10),
           xticklabels=list(map(str,range(10))) + (["Other"] if outputs > 10 else []),
           xlabel="Predicted digit (Other = outputs 10-99)",ylabel="True digit",
           title=f"Reloaded checkpoint: validation confusion matrix ({seen-correct} errors / {seen})")
    fig.tight_layout(); fig.savefig(output / "confusion_matrix.png",dpi=150); plt.close(fig)
    fig, axes = plt.subplots(4,5,figsize=(10,8))
    for ax in axes.flat: ax.axis("off")
    for ax, (image,label,pred,row) in zip(axes.flat,errors):
        ax.imshow(image.numpy(),cmap="gray")
        ax.set_title(f"train:{row}\nTrue {label}; predicted {pred}",fontsize=9)
    fig.suptitle("First 20 validation mistakes in saved membership order (not cherry-picked)")
    fig.tight_layout(); fig.savefig(output / "misclassified_examples.png",dpi=150); plt.close(fig)
    print(json.dumps({key: value for key, value in report.items()
                      if key != "confusion_matrix"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--output",required=True,help="New audit directory")
    args = parser.parse_args()
    audit(args.run,args.output)
