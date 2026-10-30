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

class CIFAR100OnlineStream(Dataset):
    """Class 0-19 stream using explicit, shared source-row membership.

    membership is a mapping (or JSON path) with source_split="train" and
    train/validation lists of original integer row IDs. No split is generated.
    seed controls training order only; validation keeps its saved order.
    """

    def __init__(self, split="train", seed=42, *, membership=None):
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        if membership is None:
            raise ValueError(
                "Saved shared membership is required: provide Nguyen's original "
                "9000 train / 1000 validation source-row IDs via membership. "
                "No replacement split will be generated."
            )
        if isinstance(membership, (str, Path)):
            with open(membership, encoding="utf-8") as handle:
                membership = json.load(handle)
        if not isinstance(membership, dict) or membership.get("source_split") != "train":
            raise ValueError("membership must declare source_split='train'")
        for name, size in (("train", 9000), ("validation", 1000)):
            rows = membership.get(name)
            if not isinstance(rows, list) or len(rows) != size:
                raise ValueError(f"{name} membership must contain {size} row IDs")
            if any(type(row) is not int or row < 0 for row in rows):
                raise ValueError(f"{name} row IDs must be nonnegative integers")
            if len(set(rows)) != size:
                raise ValueError(f"{name} membership contains duplicate row IDs")
        train_rows = set(membership["train"])
        validation_rows = set(membership["validation"])
        if train_rows & validation_rows:
            raise ValueError("train and validation membership overlap")

        self.source_split = membership["source_split"]
        self.dataset = load_dataset(
            "uoft-cs/cifar100", split=self.source_split, cache_dir=str(DATA_PATH)
        )
        selected = {
            i for i, label in enumerate(self.dataset["fine_label"])
            if label in range(20)
        }
        if len(selected) != 10000 or train_rows | validation_rows != selected:
            raise ValueError("membership must cover exactly all 10000 class 0-19 source rows")
        self.indices = list(membership[split])
        if split == "train":
            generator = torch.Generator().manual_seed(seed)
            order = torch.randperm(len(self.indices), generator=generator).tolist()
            self.indices = [self.indices[i] for i in order]
        self.transform = get_transform()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        item = self.dataset[row]
        return self.transform(item["img"]), item["fine_label"], f"{self.source_split}:{row}"


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

def get_online_loader(split="train", batch_size=32, seed=42, *, membership=None):
    """
    Return the shared stream; membership is required and never regenerated.
    """

    dataset = CIFAR100OnlineStream(
        split=split,
        seed=seed,
        membership=membership,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )