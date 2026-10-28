"""MNIST stream: fixed membership, one shuffle, source-row IDs."""
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from src.data import DATA_PATH


class MNISTStream(Dataset):
    def __init__(self, split="train", *, split_seed=42, stream_seed=42,
                 validation_size=6000, download=False):
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        self.source_split = "test" if split == "test" else "train"
        self.source = MNIST(DATA_PATH, train=split != "test", download=download,
                            transform=ToTensor())
        if split == "test":
            self.indices = list(range(len(self.source)))
        else:
            if not 0 < validation_size < len(self.source):
                raise ValueError("validation_size must leave nonempty train and validation sets")
            membership = torch.randperm(
                len(self.source), generator=torch.Generator().manual_seed(split_seed)
            ).tolist()
            self.indices = (membership[:validation_size] if split == "validation"
                            else membership[validation_size:])
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


def get_mnist_loader(split="train", batch_size=32, **kwargs):
    return DataLoader(MNISTStream(split, **kwargs), batch_size=batch_size,
                      shuffle=False, drop_last=False)
