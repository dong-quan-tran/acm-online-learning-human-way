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


def _validate_shared_c20_membership(
    membership: Union[dict, str, Path],
) -> Dict[str, Any]:
    """Load and validate the 9,000 train / 1,000 validation CIFAR-100-20 membership."""
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
            membership = _validate_shared_c20_membership(membership)

            self.dataset = (
                dataset
                if dataset is not None
                else load_dataset(
                    "uoft-cs/cifar100", split=self.source_split, cache_dir=str(DATA_PATH)
                )
            )
            self.indices = list(membership[split])
            if split == "train" and seed is not None:
                generator = torch.Generator().manual_seed(seed)
                order = torch.randperm(len(self.indices), generator=generator).tolist()
                self.indices = [self.indices[i] for i in order]
        else:
            self.source_split = "test"
            self.dataset = (
                dataset
                if dataset is not None
                else load_dataset(
                    "uoft-cs/cifar100", split=self.source_split, cache_dir=str(DATA_PATH)
                )
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


# ==============================================================================
# Periodic Held-Out Split Helpers (Adaptation Stream vs. Fixed Held-Out Set)
# ==============================================================================

_ADAPTATION_SPLIT_KEYS: Tuple[str, ...] = (
    "adaptation_ids",
    "adaptation",
    "adaptation_rows",
    "adapt_ids",
    "adapt",
    "stream_ids",
    "stream",
)
_HELD_OUT_SPLIT_KEYS: Tuple[str, ...] = (
    "held_out_ids",
    "held_out",
    "held_out_rows",
    "evaluation_ids",
    "evaluation",
    "eval_ids",
    "eval",
    "test_ids",
    "test",
)
_SINGLE_SPLIT_FALLBACK_KEYS: Tuple[str, ...] = ("indices", "rows", "ids")


def parse_source_id(
    entry: Union[int, str],
    default_source_split: str = "test",
) -> Tuple[str, int]:
    """Parse an integer row or '<split>:<row>' identifier into (split, row)."""
    if default_source_split not in {"train", "test"}:
        raise ValueError(
            f"default_source_split must be 'train' or 'test', got {default_source_split!r}"
        )
    if isinstance(entry, int) and not isinstance(entry, bool):
        if entry < 0:
            raise ValueError(f"Row ID must be non-negative, got {entry}")
        return (default_source_split, entry)
    if isinstance(entry, str) and ":" in entry:
        src_split, row_str = entry.split(":", 1)
        if src_split not in {"train", "test"} or not row_str.isdigit():
            raise ValueError(f"Malformed source ID string: {entry!r}")
        return (src_split, int(row_str))
    raise ValueError(
        f"Expected non-negative int or '<train|test>:<row>' string, got {entry!r}"
    )


def _read_json_mapping_or_list(source: Union[str, Path, Dict[str, Any], List[Any]]) -> Any:
    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return source


def load_exp_b_split_ids(
    split_source: Union[str, Path, Dict[str, Any], List[Any]],
    *,
    key: Optional[str] = None,
    split_role: Optional[str] = None,
    allow_fallback_keys: bool = True,
    default_source_split: str = "test",
) -> Tuple[List[Tuple[str, int]], Dict[str, Any]]:
    """Load an ordered list of canonical (source_split, row_id) references.

    Supports bare lists, single-role split JSONs, or combined split manifests.
    Preserves the saved element order and rejects empty or duplicate IDs.
    """
    if default_source_split not in {"train", "test"}:
        raise ValueError(
            f"default_source_split must be 'train' or 'test', got {default_source_split!r}"
        )
    raw = _read_json_mapping_or_list(split_source)
    metadata: Dict[str, Any] = {}

    if isinstance(raw, list):
        entries = raw
        resolved_default = default_source_split
        resolved_key = "<list>"
    elif isinstance(raw, dict):
        role_split_key = f"{split_role}_source_split" if split_role else None
        if role_split_key and role_split_key in raw and raw[role_split_key] is not None:
            explicit_split = raw[role_split_key]
        elif "source_split" in raw and raw["source_split"] is not None:
            explicit_split = raw["source_split"]
        else:
            explicit_split = default_source_split
        if not isinstance(explicit_split, str) or explicit_split not in {"train", "test"}:
            raise ValueError(
                f"Unsupported source_split {explicit_split!r}; expected 'train' or 'test'"
            )
        resolved_default = explicit_split

        if key is not None:
            if key not in raw:
                raise KeyError(f"Requested split key {key!r} not found in split file")
            resolved_key = key
            entries = raw[key]
        else:
            if split_role == "adaptation":
                candidates = _ADAPTATION_SPLIT_KEYS
            elif split_role == "held_out":
                candidates = _HELD_OUT_SPLIT_KEYS
            else:
                candidates = _ADAPTATION_SPLIT_KEYS + _HELD_OUT_SPLIT_KEYS
            if allow_fallback_keys:
                candidates = candidates + _SINGLE_SPLIT_FALLBACK_KEYS

            resolved_key = next((k for k in candidates if k in raw), None)
            if resolved_key is None:
                raise KeyError(
                    f"Could not locate {split_role or 'split'} IDs in split mapping; "
                    f"available keys: {sorted(raw.keys())}"
                )
            entries = raw[resolved_key]

        metadata = {
            k: v
            for k, v in raw.items()
            if not isinstance(v, list)
        }
    else:
        raise TypeError(f"Expected JSON object or list, got {type(raw).__name__}")

    if not isinstance(entries, list) or len(entries) == 0:
        raise ValueError(
            f"Split {split_role or resolved_key!r} must be a non-empty list of IDs"
        )

    refs = [parse_source_id(item, default_source_split=resolved_default) for item in entries]
    if len(set(refs)) != len(refs):
        raise ValueError(
            f"Split {split_role or resolved_key!r} contains duplicate source IDs"
        )
    metadata["resolved_key"] = resolved_key
    return refs, metadata


def load_exp_b_split_pair(
    split_file: Union[str, Path, Dict[str, Any]],
    held_out_file: Optional[Union[str, Path, Dict[str, Any]]] = None,
    *,
    adaptation_key: Optional[str] = None,
    held_out_key: Optional[str] = None,
    train_val_membership: Optional[Union[str, Path, Dict[str, Any]]] = SHARED_C20_MEMBERSHIP_PATH,
    prior_test_membership: Optional[Union[str, Path, Dict[str, Any]]] = EXP_B_TEST_MEMBERSHIP_PATH,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], Dict[str, Any]]:
    """Load disjoint adaptation and held-out splits with full overlap checks.

    Verifies:
    - Neither adaptation nor held-out is empty or has internal duplicates.
    - Adaptation and held-out are strictly disjoint.
    - Neither overlaps source training or development/validation splits.
    - Records whether the held-out images overlap prior Experiment B test runs
      (data/exp_b_test_membership.json) so reports do not mislabel them as unseen.
    """
    separate_files = held_out_file is not None
    raw_split = _read_json_mapping_or_list(split_file)
    raw_held_out = _read_json_mapping_or_list(held_out_file) if separate_files else raw_split

    adapt_refs, adapt_meta = load_exp_b_split_ids(
        raw_split,
        key=adaptation_key,
        split_role="adaptation",
        allow_fallback_keys=separate_files,
    )
    held_out_refs, held_out_meta = load_exp_b_split_ids(
        raw_held_out,
        key=held_out_key,
        split_role="held_out",
        allow_fallback_keys=separate_files,
    )

    adapt_set = set(adapt_refs)
    held_out_set = set(held_out_refs)
    overlap_adapt_held = adapt_set & held_out_set
    if overlap_adapt_held:
        sample = sorted(overlap_adapt_held)[:5]
        raise ValueError(
            f"Adaptation and held-out splits overlap ({len(overlap_adapt_held)} shared IDs, "
            f"first: {sample})"
        )

    embedded_train_set: Optional[set] = None
    embedded_dev_set: Optional[set] = None
    if isinstance(raw_split, dict):
        if raw_split.get("source_train_ids") is not None:
            raw_st = raw_split["source_train_ids"]
            if not isinstance(raw_st, list) or not raw_st:
                raise ValueError("Embedded 'source_train_ids' must be a non-empty list")
            st_parsed = [parse_source_id(r, default_source_split="train") for r in raw_st]
            if len(set(st_parsed)) != len(st_parsed):
                raise ValueError("Embedded 'source_train_ids' contains duplicate source IDs")
            embedded_train_set = set(st_parsed)
        if raw_split.get("development_ids") is not None:
            raw_dev = raw_split["development_ids"]
            if not isinstance(raw_dev, list) or not raw_dev:
                raise ValueError("Embedded 'development_ids' must be a non-empty list")
            dev_parsed = [parse_source_id(r, default_source_split="train") for r in raw_dev]
            if len(set(dev_parsed)) != len(dev_parsed):
                raise ValueError("Embedded 'development_ids' contains duplicate source IDs")
            embedded_dev_set = set(dev_parsed)

    embedded_source_lists_verified = False
    if embedded_train_set is not None and embedded_dev_set is not None:
        if embedded_train_set & embedded_dev_set:
            raise ValueError(
                f"Embedded 'source_train_ids' and 'development_ids' overlap "
                f"({len(embedded_train_set & embedded_dev_set)} shared IDs)"
            )
    embedded_pool = (embedded_train_set or set()) | (embedded_dev_set or set())
    if embedded_pool:
        if adapt_set & embedded_pool:
            raise ValueError(
                f"Adaptation split overlaps embedded source_train_ids/development_ids "
                f"({len(adapt_set & embedded_pool)} shared IDs)"
            )
        if held_out_set & embedded_pool:
            raise ValueError(
                f"Held-out split overlaps embedded source_train_ids/development_ids "
                f"({len(held_out_set & embedded_pool)} shared IDs)"
            )
        embedded_source_lists_verified = True

    if train_val_membership is not None:
        tv_raw = _read_json_mapping_or_list(train_val_membership)
        if not isinstance(tv_raw, dict):
            raise TypeError(
                f"train_val_membership must be a JSON object, got {type(tv_raw).__name__}"
            )
        tv_split = tv_raw.get("source_split", "train")
        if not isinstance(tv_split, str) or tv_split not in {"train", "test"}:
            raise ValueError(
                f"Unsupported train_val_membership source_split {tv_split!r}; expected 'train' or 'test'"
            )
        if "train" not in tv_raw or "validation" not in tv_raw:
            raise ValueError("train_val_membership must include 'train' and 'validation' lists")
        train_entries = tv_raw["train"]
        val_entries = tv_raw["validation"]
        if not isinstance(train_entries, list) or not isinstance(val_entries, list):
            raise ValueError("train_val_membership 'train' and 'validation' must be lists")
        train_refs = {parse_source_id(r, default_source_split=tv_split) for r in train_entries}
        val_refs = {parse_source_id(r, default_source_split=tv_split) for r in val_entries}
        if embedded_train_set is not None and embedded_train_set != train_refs:
            raise ValueError(
                "Embedded 'source_train_ids' in split file does not match train_val_membership['train']"
            )
        if embedded_dev_set is not None and embedded_dev_set != val_refs:
            raise ValueError(
                "Embedded 'development_ids' in split file does not match train_val_membership['validation']"
            )
        source_pool = train_refs | val_refs
        if adapt_set & source_pool:
            raise ValueError(
                f"Adaptation split overlaps source train/validation split "
                f"({len(adapt_set & source_pool)} shared IDs)"
            )
        if held_out_set & source_pool:
            raise ValueError(
                f"Held-out split overlaps source train/validation split "
                f"({len(held_out_set & source_pool)} shared IDs)"
            )

    held_out_prior_overlap = 0
    adapt_prior_overlap = 0
    if prior_test_membership is not None:
        pt_path = Path(prior_test_membership) if isinstance(prior_test_membership, (str, Path)) else None
        if (
            pt_path is not None
            and not pt_path.exists()
            and pt_path.resolve() != EXP_B_TEST_MEMBERSHIP_PATH.resolve()
        ):
            raise FileNotFoundError(f"prior_test_membership file not found: {pt_path}")
        if pt_path is None or pt_path.exists():
            pt_raw = _read_json_mapping_or_list(prior_test_membership)
            if not isinstance(pt_raw, dict):
                raise TypeError(
                    f"prior_test_membership must be a JSON object, got {type(pt_raw).__name__}"
                )
            pt_split = pt_raw.get("source_split", "test")
            if not isinstance(pt_split, str) or pt_split not in {"train", "test"}:
                raise ValueError(
                    f"Unsupported prior_test_membership source_split {pt_split!r}; expected 'train' or 'test'"
                )
            if "test" in pt_raw:
                pt_entries = pt_raw["test"]
            elif "held_out" in pt_raw:
                pt_entries = pt_raw["held_out"]
            else:
                raise ValueError("prior_test_membership must include a 'test' or 'held_out' list")
            if not isinstance(pt_entries, list):
                raise ValueError("prior_test_membership entries must be a list")
            prior_refs = {parse_source_id(r, default_source_split=pt_split) for r in pt_entries}
            held_out_prior_overlap = len(held_out_set & prior_refs)
            adapt_prior_overlap = len(adapt_set & prior_refs)

    metadata: Dict[str, Any] = {
        "adaptation_count": len(adapt_refs),
        "held_out_count": len(held_out_refs),
        "adaptation_key": adapt_meta.get("resolved_key"),
        "held_out_key": held_out_meta.get("resolved_key"),
        "adaptation_metadata": adapt_meta,
        "held_out_metadata": held_out_meta,
        "embedded_source_lists_verified": embedded_source_lists_verified,
        "held_out_examined_in_prior_exp_b": held_out_prior_overlap > 0,
        "held_out_prior_overlap_count": held_out_prior_overlap,
        "adaptation_prior_overlap_count": adapt_prior_overlap,
    }
    return adapt_refs, held_out_refs, metadata


class CIFAR100SplitSubsetDataset(Dataset):
    """Ordered CIFAR-100 classes 0-19 dataset backed by explicit (split, row) IDs.

    Preserves the exact sample order in ``sample_refs``, applies pixel-space [0, 1]
    ``ToTensor()`` preprocessing, validates that all labels belong to classes 0..19,
    and supports deterministic per-sample SHA-256 Gaussian noise so clean and noisy
    views stay aligned by ``sample_id = f"{source_split}:{row}"``.
    """

    def __init__(
        self,
        sample_refs: List[Tuple[str, int]],
        *,
        corruption: str = "clean",
        noise_sigma: float = 0.08,
        noise_seed: int = 42,
        datasets_by_split: Optional[Dict[str, Any]] = None,
        allowed_classes: Tuple[int, ...] = tuple(range(20)),
    ):
        if corruption not in {"clean", "gaussian_noise"}:
            raise ValueError("corruption must be 'clean' or 'gaussian_noise'")
        if not sample_refs:
            raise ValueError("sample_refs must be non-empty")
        if len(set(sample_refs)) != len(sample_refs):
            raise ValueError("sample_refs contains duplicate entries")

        self.sample_refs = list(sample_refs)
        self.corruption = corruption
        self.noise_sigma = float(noise_sigma)
        self.noise_seed = int(noise_seed)
        self.allowed_classes = set(allowed_classes)

        needed_splits = sorted({src_split for src_split, _ in self.sample_refs})
        self.datasets_by_split: Dict[str, Any] = {}
        for src_split in needed_splits:
            if src_split not in {"train", "test"}:
                raise ValueError(f"Unsupported CIFAR-100 source split {src_split!r}")
            if datasets_by_split is not None and datasets_by_split.get(src_split) is not None:
                ds = datasets_by_split[src_split]
            else:
                ds = load_dataset(
                    "uoft-cs/cifar100", split=src_split, cache_dir=str(DATA_PATH)
                )
            self.datasets_by_split[src_split] = ds

        # Validate index bounds and class membership (0..19) upfront
        labels_cache: Dict[str, Any] = {}
        for src_split, ds in self.datasets_by_split.items():
            try:
                labels_cache[src_split] = ds["fine_label"]
            except (KeyError, TypeError, IndexError, ValueError):
                labels_cache[src_split] = None

        for src_split, row in self.sample_refs:
            ds = self.datasets_by_split[src_split]
            if row < 0 or row >= len(ds):
                raise IndexError(
                    f"Row {row} out of range for split '{src_split}' (len={len(ds)})"
                )
            cached_labels = labels_cache[src_split]
            lbl = int(cached_labels[row]) if cached_labels is not None else int(ds[row]["fine_label"])
            if lbl not in self.allowed_classes:
                raise ValueError(
                    f"Sample '{src_split}:{row}' has fine_label={lbl} outside allowed classes"
                )

        self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.sample_refs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        src_split, row = self.sample_refs[index]
        item = self.datasets_by_split[src_split][row]
        image = self.transform(item["img"])
        label = int(item["fine_label"])
        sample_id = f"{src_split}:{row}"

        if self.corruption == "gaussian_noise":
            image = apply_gaussian_noise(
                image, sample_id, sigma=self.noise_sigma, noise_seed=self.noise_seed
            )

        return image, label, sample_id


def get_exp_b_split_loaders(
    split_file: Union[str, Path, Dict[str, Any]],
    held_out_file: Optional[Union[str, Path, Dict[str, Any]]] = None,
    *,
    adaptation_key: Optional[str] = None,
    held_out_key: Optional[str] = None,
    stream_corruption: str = "clean",
    noise_sigma: float = 0.08,
    noise_seed: int = 42,
    eval_batch_size: int = 64,
    num_workers: int = 0,
    max_adapt_images: Optional[int] = None,
    datasets_by_split: Optional[Dict[str, Any]] = None,
    train_val_membership: Optional[Union[str, Path, Dict[str, Any]]] = SHARED_C20_MEMBERSHIP_PATH,
    prior_test_membership: Optional[Union[str, Path, Dict[str, Any]]] = EXP_B_TEST_MEMBERSHIP_PATH,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """Build adaptation, clean held-out, and noisy held-out DataLoaders from split files.

    Returns ``(adaptation_loader, held_out_clean_loader, held_out_noisy_loader, metadata)``.
    Adaptation always keeps ``batch_size=1`` and ``shuffle=False`` in saved stream order.
    """
    if max_adapt_images is not None and max_adapt_images <= 0:
        raise ValueError(f"max_adapt_images must be positive, got {max_adapt_images}")

    adapt_refs, held_out_refs, metadata = load_exp_b_split_pair(
        split_file,
        held_out_file,
        adaptation_key=adaptation_key,
        held_out_key=held_out_key,
        train_val_membership=train_val_membership,
        prior_test_membership=prior_test_membership,
    )
    if max_adapt_images is not None:
        adapt_refs = adapt_refs[:max_adapt_images]
        metadata["adaptation_count_effective"] = len(adapt_refs)
    else:
        metadata["adaptation_count_effective"] = len(adapt_refs)

    adapt_ds = CIFAR100SplitSubsetDataset(
        adapt_refs,
        corruption=stream_corruption,
        noise_sigma=noise_sigma,
        noise_seed=noise_seed,
        datasets_by_split=datasets_by_split,
    )
    # Reuse already-loaded HuggingFace datasets for held-out splits when splits match
    shared_cache = dict(datasets_by_split) if datasets_by_split is not None else {}
    shared_cache.update(adapt_ds.datasets_by_split)
    held_out_clean_ds = CIFAR100SplitSubsetDataset(
        held_out_refs,
        corruption="clean",
        noise_sigma=noise_sigma,
        noise_seed=noise_seed,
        datasets_by_split=shared_cache if shared_cache else None,
    )
    shared_cache.update(held_out_clean_ds.datasets_by_split)
    held_out_noisy_ds = CIFAR100SplitSubsetDataset(
        held_out_refs,
        corruption="gaussian_noise",
        noise_sigma=noise_sigma,
        noise_seed=noise_seed,
        datasets_by_split=shared_cache,
    )

    adapt_loader = DataLoader(
        adapt_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    held_out_clean_loader = DataLoader(
        held_out_clean_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    held_out_noisy_loader = DataLoader(
        held_out_noisy_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return adapt_loader, held_out_clean_loader, held_out_noisy_loader, metadata
