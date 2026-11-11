import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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


# ==============================================================================
# Experiment B (TTT): CIFAR-100 (Classes 0-19) Dataset, Loaders & Noise
# ==============================================================================

SHARED_C20_MEMBERSHIP_PATH = ROOT / "data" / "shared_c20_membership.json"
EXP_B_TEST_MEMBERSHIP_PATH = ROOT / "data" / "exp_b_test_membership.json"


def get_sample_noise_seed(noise_seed: int, sample_id: Union[str, int]) -> int:
    """Derive a deterministic 64-bit unsigned integer seed via SHA-256."""
    payload = f"{noise_seed}_{sample_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def apply_gaussian_noise(
    image_tensor: torch.Tensor,
    sample_id: Union[str, int, torch.Tensor, List[Union[str, int]], Tuple[Union[str, int], ...]],
    sigma: float = 0.08,
    noise_seed: int = 42,
) -> torch.Tensor:
    """Add deterministic zero-mean Gaussian noise clamped to [0.0, 1.0] in pixel space.

    Operates strictly in [0.0, 1.0] pixel space before channel mean/std normalization
    and before rotation augmentation. Seeding is uniquely derived from
    (noise_seed, sample_id) via SHA-256 so that identical noisy tensors are generated
    across all evaluation conditions (supervised_frozen, joint_frozen, standard_ttt,
    online_ttt) regardless of batch size, worker concurrency, or stream order.

    To ensure bit-for-bit reproducibility across all hardware platforms (CPU, CUDA, MPS),
    noise is generated using a dedicated CPU torch.Generator and transferred to the
    input tensor's device.
    """
    if not torch.is_floating_point(image_tensor):
        raise TypeError(
            f"image_tensor must be a floating point tensor in [0.0, 1.0] pixel space, "
            f"got {image_tensor.dtype}. Convert from PIL with transforms.ToTensor() first."
        )

    # Handle batched input when sample_id is a tensor, list, or tuple
    if isinstance(sample_id, torch.Tensor):
        if sample_id.numel() == 1 and (
            image_tensor.dim() == 3
            or (image_tensor.dim() == 4 and image_tensor.size(0) == 1)
        ):
            sample_id = sample_id.item()
        else:
            sample_id = sample_id.tolist()

    if isinstance(sample_id, (list, tuple)):
        if image_tensor.dim() != 4 or len(sample_id) != image_tensor.size(0):
            raise ValueError(
                f"Batch dimension mismatch: image_tensor has shape {image_tensor.shape} "
                f"but {len(sample_id)} sample_ids were provided"
            )
        return torch.stack([
            apply_gaussian_noise(
                image_tensor[i], sample_id[i], sigma=sigma, noise_seed=noise_seed
            )
            for i in range(len(sample_id))
        ])

    if image_tensor.dim() == 4 and image_tensor.size(0) > 1:
        raise ValueError(
            f"Batch dimension mismatch: image_tensor has shape {image_tensor.shape} "
            f"but sample_id is not a list or tuple of matching length."
        )

    if image_tensor.dim() not in (3, 4):
        raise ValueError(
            f"image_tensor must be 3D (C, H, W) or 4D (1, C, H, W), got shape {image_tensor.shape}"
        )

    seed = get_sample_noise_seed(noise_seed, sample_id)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        image_tensor.shape,
        generator=gen,
        dtype=image_tensor.dtype,
        device="cpu",
    )
    if noise.device != image_tensor.device:
        noise = noise.to(image_tensor.device)

    noisy = image_tensor + noise * sigma
    return torch.clamp(noisy, 0.0, 1.0)


def build_exp_b_test_membership(
    save_path: Optional[Union[str, Path]] = None,
    dataset: Optional[Any] = None,
) -> Dict[str, Any]:
    """Extract official CIFAR-100 test split indices for classes 0-19 in ascending order.

    If save_path is provided, serializes the membership dictionary to JSON.
    """
    if dataset is None:
        dataset = load_dataset(
            "uoft-cs/cifar100", split="test", cache_dir=str(DATA_PATH)
        )
    test_rows = [
        i for i, label in enumerate(dataset["fine_label"]) if label in range(20)
    ]
    test_rows.sort()
    membership = {
        "source_split": "test",
        "classes": list(range(20)),
        "num_samples": len(test_rows),
        "test": test_rows,
    }
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(membership, f, indent=2)
    return membership


class CIFAR100ExpBDataset(Dataset):
    """Dataset for Experiment B (CIFAR-100 classes 0-19).

    Supports:
    - 'train': 9,000 images from shared_c20_membership.json['train']
    - 'validation': 1,000 images from shared_c20_membership.json['validation']
    - 'test': 2,000 images from exp_b_test_membership.json['test'] (classes 0-19)

    Features:
    - Optional training augmentations (RandomCrop + RandomHorizontalFlip)
    - Optional deterministic Gaussian noise (sigma=0.08, clamped [0, 1])
    - Returns (image, label, sample_id) where image is [0, 1] float tensor,
      label is fine_label in 0..19, sample_id is f"{source_split}:{row_id}".
    """

    def __init__(
        self,
        split: str = "train",
        seed: Optional[int] = 42,
        *,
        membership: Optional[Union[dict, str, Path]] = None,
        test_membership: Optional[Union[dict, str, Path]] = None,
        augment: bool = False,
        transform: Optional[Any] = None,
        corruption: str = "clean",
        noise_sigma: float = 0.08,
        noise_seed: int = 42,
        dataset: Optional[Any] = None,
    ):
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be 'train', 'validation', or 'test'")
        if corruption not in {"clean", "gaussian_noise"}:
            raise ValueError("corruption must be 'clean' or 'gaussian_noise'")

        self.split = split
        self.seed = seed
        self.corruption = corruption
        self.noise_sigma = noise_sigma
        self.noise_seed = noise_seed

        if split in {"train", "validation"}:
            self.source_split = "train"
            if membership is None:
                membership = SHARED_C20_MEMBERSHIP_PATH
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

            self.dataset = dataset or load_dataset(
                "uoft-cs/cifar100", split=self.source_split, cache_dir=str(DATA_PATH)
            )
            self.indices = list(membership[split])
            if split == "train" and seed is not None:
                generator = torch.Generator().manual_seed(seed)
                order = torch.randperm(len(self.indices), generator=generator).tolist()
                self.indices = [self.indices[i] for i in order]
        else:
            self.source_split = "test"
            self.dataset = dataset or load_dataset(
                "uoft-cs/cifar100", split=self.source_split, cache_dir=str(DATA_PATH)
            )
            if test_membership is None:
                test_membership = EXP_B_TEST_MEMBERSHIP_PATH
            if isinstance(test_membership, (str, Path)):
                path = Path(test_membership)
                if path.exists():
                    with open(path, encoding="utf-8") as handle:
                        test_membership = json.load(handle)
                else:
                    test_membership = build_exp_b_test_membership(save_path=path, dataset=self.dataset)

            if not isinstance(test_membership, dict) or test_membership.get("source_split") != "test":
                raise ValueError("test_membership must declare source_split='test'")
            rows = test_membership.get("test")
            if not isinstance(rows, list) or len(rows) != 2000:
                raise ValueError("test membership must contain 2000 row IDs")
            if any(type(row) is not int or row < 0 for row in rows):
                raise ValueError("test row IDs must be nonnegative integers")
            if len(set(rows)) != 2000:
                raise ValueError("test membership contains duplicate row IDs")

            self.indices = list(rows)

        if transform is not None:
            self.transform = transform
        elif split == "train" and augment:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        row = self.indices[index]
        item = self.dataset[row]
        image = self.transform(item["img"])
        label = int(item["fine_label"])
        sample_id = f"{self.source_split}:{row}"

        if self.corruption == "gaussian_noise":
            image = apply_gaussian_noise(
                image, sample_id, sigma=self.noise_sigma, noise_seed=self.noise_seed
            )

        return image, label, sample_id


def get_exp_b_loaders(
    membership_path: Optional[Union[str, Path]] = None,
    test_membership_path: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    *,
    seed: int = 42,
    augment_train: bool = True,
    val_batch_size: Optional[int] = None,
    test_batch_size: Optional[int] = None,
    corruption: str = "clean",
    noise_sigma: float = 0.08,
    noise_seed: int = 42,
    shuffle_train: bool = True,
    dataset_train: Optional[Any] = None,
    dataset_test: Optional[Any] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) for Experiment B."""
    if membership_path is None:
        membership_path = SHARED_C20_MEMBERSHIP_PATH
    if test_membership_path is None:
        test_membership_path = EXP_B_TEST_MEMBERSHIP_PATH

    train_dataset = CIFAR100ExpBDataset(
        split="train",
        seed=seed,
        membership=membership_path,
        augment=augment_train,
        dataset=dataset_train,
    )
    val_dataset = CIFAR100ExpBDataset(
        split="validation",
        membership=membership_path,
        augment=False,
        dataset=dataset_train,
    )
    test_dataset = CIFAR100ExpBDataset(
        split="test",
        test_membership=test_membership_path,
        augment=False,
        corruption=corruption,
        noise_sigma=noise_sigma,
        noise_seed=noise_seed,
        dataset=dataset_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size if val_batch_size is not None else batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size if test_batch_size is not None else batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def get_exp_b_loader(
    split: str = "test",
    batch_size: int = 1,
    num_workers: int = 0,
    *,
    seed: int = 42,
    augment: bool = False,
    corruption: str = "clean",
    noise_sigma: float = 0.08,
    noise_seed: int = 42,
    shuffle: Optional[bool] = None,
    membership_path: Optional[Union[str, Path]] = None,
    test_membership_path: Optional[Union[str, Path]] = None,
    dataset: Optional[Any] = None,
) -> DataLoader:
    """Return a single DataLoader for Experiment B."""
    if membership_path is None:
        membership_path = SHARED_C20_MEMBERSHIP_PATH
    if test_membership_path is None:
        test_membership_path = EXP_B_TEST_MEMBERSHIP_PATH

    dataset_inst = CIFAR100ExpBDataset(
        split=split,
        seed=seed,
        membership=membership_path if split in {"train", "validation"} else None,
        test_membership=test_membership_path if split == "test" else None,
        augment=augment,
        corruption=corruption,
        noise_sigma=noise_sigma,
        noise_seed=noise_seed,
        dataset=dataset,
    )
    if shuffle is None:
        shuffle = (split == "train")

    return DataLoader(
        dataset_inst,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
