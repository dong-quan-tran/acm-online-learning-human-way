"""Real cached CIFAR-100 checks with TEST-ONLY membership, not Nguyen's split."""
import copy
import json

import pytest
import torch
from torch.utils.data import SequentialSampler

from src import data


@pytest.fixture(scope="module")
def source():
    return data.load_dataset("uoft-cs/cifar100", split="train", cache_dir=str(data.DATA_PATH))


@pytest.fixture
def membership(source):
    # Test fixture only. Never saved as or presented as the team's membership.
    rows = [i for i, label in enumerate(source["fine_label"]) if label in range(20)]
    return {"source_split": "train", "train": rows[:9000], "validation": rows[9000:]}


@pytest.fixture
def loaders(monkeypatch, source, membership):
    monkeypatch.setattr(data, "load_dataset", lambda *args, **kwargs: source)
    return lambda split="train", seed=42: data.get_online_loader(
        split=split, seed=seed, membership=membership
    )


def test_saved_membership_required():
    with pytest.raises(ValueError, match="Saved shared membership is required"):
        data.get_online_loader()


def test_shared_membership_and_coverage(loaders, membership, source):
    train, validation = loaders(), loaders("validation")
    train_rows, val_rows = set(train.dataset.indices), set(validation.dataset.indices)
    intended = {i for i, label in enumerate(source["fine_label"]) if label in range(20)}
    assert len(train.dataset) == len(train_rows) == 9000
    assert len(validation.dataset) == len(val_rows) == 1000
    assert not train_rows & val_rows
    assert train_rows | val_rows == intended
    assert len(intended) == 10000
    assert train_rows == set(membership["train"])
    assert validation.dataset.indices == membership["validation"]
    assert {source[i]["fine_label"] for i in train_rows | val_rows} == set(range(20))


def test_seed_changes_order_only(loaders):
    first, repeat, other = loaders(), loaders(), loaders(seed=43)
    assert first.dataset.indices == repeat.dataset.indices
    assert first.dataset.indices != other.dataset.indices
    assert set(first.dataset.indices) == set(other.dataset.indices)
    assert loaders("validation").dataset.indices == loaders("validation", seed=43).dataset.indices
    assert isinstance(first.sampler, SequentialSampler)


def test_source_row_ids_and_scaling(loaders, source):
    for split in ("train", "validation"):
        dataset = loaders(split).dataset
        for position in (0, len(dataset) - 1):
            image, label, sample_id = dataset[position]
            row = dataset.indices[position]
            assert sample_id == f"train:{row}"
            assert label == source[row]["fine_label"]
            assert torch.equal(image, data.get_transform()(source[row]["img"]))
            assert 0 <= image.min() <= image.max() <= 1


def test_batches_and_once_only_coverage(loaders):
    loader = loaders()
    assert loader.batch_size == 32
    assert loader.drop_last is False
    assert len(loader) == 282
    ids, labels, sizes = [], [], []
    for images, batch_labels, sample_ids in loader:
        assert images.shape == (len(batch_labels), 3, 32, 32)
        assert len(sample_ids) == len(batch_labels)
        ids.extend(sample_ids)
        labels.extend(batch_labels.tolist())
        sizes.append(len(images))
    assert sizes[:-1] == [32] * 281
    assert sizes[-1] == 8
    assert sum(sizes) == len(ids) == len(set(ids)) == 9000
    assert ids == [f"train:{row}" for row in loader.dataset.indices]
    assert set(labels) == set(range(20))


def test_batch_consumer_integration(loaders):
    # Contract smoke check, not a claim of integration with unpublished training code.
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 100))
    images, labels, sample_ids = next(iter(loaders()))
    loss = torch.nn.functional.cross_entropy(model(images), labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert len(sample_ids) == 32
    with torch.no_grad():
        images, labels, sample_ids = next(iter(loaders("validation")))
        assert model(images).shape == (32, 100)
        assert labels.shape == (32,)
        assert len(sample_ids) == 32


def test_membership_json_consumed_exactly(tmp_path, monkeypatch, source, membership):
    monkeypatch.setattr(data, "load_dataset", lambda *args, **kwargs: source)
    path = tmp_path / "membership.json"
    path.write_text(json.dumps(membership))
    loader = data.get_online_loader("validation", membership=path)
    assert loader.dataset.indices == membership["validation"]


@pytest.mark.parametrize("problem", ["size", "duplicate", "overlap", "wrong_rows", "source", "row_type"])
def test_invalid_membership_rejected(problem, monkeypatch, source, membership):
    monkeypatch.setattr(data, "load_dataset", lambda *args, **kwargs: source)
    invalid = copy.deepcopy(membership)
    if problem == "size":
        invalid["train"].pop()
    elif problem == "duplicate":
        invalid["train"][0] = invalid["train"][1]
    elif problem == "overlap":
        invalid["validation"][0] = invalid["train"][0]
    elif problem == "wrong_rows":
        invalid["train"][0] = len(source) + 1
    elif problem == "source":
        invalid["source_split"] = "test"
    else:
        invalid["train"][0] = "0"
    with pytest.raises(ValueError):
        data.get_online_loader(membership=invalid)
