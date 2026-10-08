import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
TASK_SPLIT_PATH = ROOT / "data" / "task_split.json"
DATA_PATH = ROOT / "data" / "raw"


def get_transform():
    """Basic preprocessing for CIFAR-100."""
    return transforms.ToTensor()


def get_task_loader(task_id, split="train", batch_size=64, seed=42):
    
    """
    Return a DataLoader for one CIFAR-100 task.

    Example:
        task_id=1 -> classes 0-9
        task_id=2 -> classes 10-19
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    # Read task definitions
    with open(TASK_SPLIT_PATH, "r") as f:
        task_split = json.load(f)

    task_name = f"task_{task_id}"

    if task_name not in task_split:
        raise ValueError(f"Unknown task: {task_name}")

    class_ids = set(task_split[task_name]["class_ids"])

    # Load CIFAR-100
    dataset = datasets.CIFAR100(
        root=DATA_PATH,
        train=(split == "train"),
        download=True,
        transform=get_transform(),
    )

    # Keep only images belonging to this task
    indices = [
        i for i, label in enumerate(dataset.targets)
        if label in class_ids
    ]

    task_dataset = Subset(dataset, indices)

    # Reproducible shuffle
    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        task_dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        generator=generator,
    )


