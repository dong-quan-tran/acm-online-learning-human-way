import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
TASK_SPLIT_PATH = ROOT / "data" / "task_split.json"
DATA_PATH = ROOT / "data" / "raw"


def get_transform():
    """Basic preprocessing for CIFAR-100."""
    return transforms.ToTensor()


class CIFAR100Task(Dataset):
    """
    CIFAR-100 subset for one continual-learning task.
    """

    def __init__(self, task_id, split="train"):
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        # Read task definitions
        with open(TASK_SPLIT_PATH, "r") as f:
            task_split = json.load(f)

        task_name = f"task_{task_id}"

        if task_name not in task_split:
            raise ValueError(f"Unknown task: {task_name}")

        class_ids = set(task_split[task_name]["class_ids"])

        # Load CIFAR-100 from Hugging Face
        self.dataset = load_dataset(
            "uoft-cs/cifar100",
            split=split,
            cache_dir=str(DATA_PATH),
        )

        # Keep only images belonging to this task
        self.indices = [
            i
            for i, label in enumerate(self.dataset["fine_label"])
            if label in class_ids
        ]

        self.transform = get_transform()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        item = self.dataset[self.indices[index]]

        image = self.transform(item["img"])
        label = item["fine_label"]

        return image, label


def get_task_loader(task_id, split="train", batch_size=64, seed=42):
    """
    Return a DataLoader for one CIFAR-100 task.

    Example:
        task_id=1 -> classes 0-9
        task_id=2 -> classes 10-19
    """

    dataset = CIFAR100Task(
        task_id=task_id,
        split=split,
    )

    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        generator=generator,
    )