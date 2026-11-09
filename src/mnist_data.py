"""MNIST stream: fixed membership, one shuffle, source-row IDs."""
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToTensor, v2

from src.data import DATA_PATH


def get_mnist_transform(profile="native", training=False):
    if profile == "native":
        return ToTensor()
    if profile != "nguyen_literal":
        raise ValueError("unknown MNIST preprocessing profile")
    # Adapt source images once; match Nguyen's uint8 tensor transform thereafter.
    steps = [transforms.Resize((32, 32), antialias=True),
             transforms.Grayscale(num_output_channels=3), transforms.PILToTensor()]
    if training:
        steps += [v2.RandomCrop(32, padding=4, padding_mode="reflect"),
                  v2.RandomHorizontalFlip()]
    return transforms.Compose(steps + [v2.ToDtype(torch.float32, scale=True)])


class MNISTStream(Dataset):
    def __init__(self, split="train", *, split_seed=42, stream_seed=42,
                 validation_size=6000, train_size=None, download=False,
                 profile="native", membership_strategy="random"):
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        self.source_split = "test" if split == "test" else "train"
        self.source = MNIST(DATA_PATH, train=split != "test", download=download,
                            transform=get_mnist_transform(profile, training=split == "train"))
        if split == "test":
            self.indices = list(range(len(self.source)))
        else:
            if not 0 < validation_size < len(self.source):
                raise ValueError("validation_size must leave nonempty train and validation sets")
            membership = torch.randperm(
                len(self.source), generator=torch.Generator().manual_seed(split_seed)
            ).tolist()
            available = len(self.source) - validation_size
            if train_size is None:
                train_size = available
            if type(train_size) is not int or not 0 < train_size <= available:
                raise ValueError("train_size must fit the remaining training-source rows")
            if membership_strategy == "random":
                self.indices = (membership[:validation_size] if split == "validation"
                                else membership[validation_size:validation_size + train_size])
            elif membership_strategy == "nguyen_stratified":
                if train_size % 10 or validation_size % 10:
                    raise ValueError("stratified sizes must divide equally across ten digits")
                rng = np.random.default_rng(split_seed)
                labels = np.asarray(self.source.targets)
                train_rows, val_rows = [], []
                for digit in range(10):
                    rows = rng.permutation(np.flatnonzero(labels == digit))
                    n_val, n_train = validation_size // 10, train_size // 10
                    if len(rows) < n_val + n_train:
                        raise ValueError("not enough source rows in digit class")
                    val_rows.extend(rows[:n_val].tolist())
                    train_rows.extend(rows[n_val:n_val+n_train].tolist())
                self.indices = sorted(train_rows if split == "train" else val_rows)
            else:
                raise ValueError("unknown membership strategy")
            if split == "train":
                order = torch.randperm(len(self.indices), generator=
                                      torch.Generator().manual_seed(stream_seed)).tolist()
                self.indices = [self.indices[i] for i in order]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        row = self.indices[position]
        image, label = self.source[row]
        return image, label, f"{self.source_split}:{row}"


def get_mnist_loader(split="train", batch_size=32, num_workers=0,
                     seed_workers=False, **kwargs):
    generator = (torch.Generator().manual_seed(kwargs.get("stream_seed", 42))
                 if seed_workers else None)
    return DataLoader(MNISTStream(split, **kwargs), batch_size=batch_size,
                      shuffle=False, drop_last=False, num_workers=num_workers,
                      generator=generator)
