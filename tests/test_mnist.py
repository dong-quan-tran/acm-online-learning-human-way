"""Offline contract tests; fake images do not establish MNIST accuracy."""
import pytest
import torch
from torch.utils.data import SequentialSampler
from PIL import Image

from src import mnist_data
from src.models import SmallResNet


class FakeMNIST:
    def __init__(self, root, train, download, transform):
        self.size = 60000 if train else 10000
        self.transform = transform
        self.targets = torch.arange(self.size) % 10

    def __len__(self):
        return self.size

    def __getitem__(self, row):
        return self.transform(Image.new("L", (28, 28), color=row % 256)), row % 10


@pytest.fixture(autouse=True)
def fake_source(monkeypatch):
    monkeypatch.setattr(mnist_data, "MNIST", FakeMNIST)


def test_mnist_membership_and_order():
    train = mnist_data.MNISTStream()
    validation = mnist_data.MNISTStream("validation")
    repeat = mnist_data.MNISTStream()
    changed = mnist_data.MNISTStream(stream_seed=43)
    assert len(train) == len(set(train.indices)) == 54000
    assert len(validation) == len(set(validation.indices)) == 6000
    assert not set(train.indices) & set(validation.indices)
    assert set(train.indices) | set(validation.indices) == set(range(60000))
    assert train.indices == repeat.indices
    assert train.indices != changed.indices
    assert set(train.indices) == set(changed.indices)
    assert validation.indices == mnist_data.MNISTStream("validation", stream_seed=43).indices


def test_mnist_labels_shapes_ids_and_coverage():
    loader = mnist_data.get_mnist_loader()
    assert isinstance(loader.sampler, SequentialSampler)
    assert loader.batch_size == 32 and not loader.drop_last
    ids, labels = [], set()
    for images, batch_labels, sample_ids in loader:
        assert images.shape == (len(batch_labels), 1, 28, 28)
        assert 0 <= images.min() <= images.max() <= 1
        ids.extend(sample_ids)
        labels.update(batch_labels.tolist())
    assert len(loader) == 1688
    assert len(batch_labels) == 16
    assert len(ids) == len(set(ids)) == 54000
    assert ids == [f"train:{row}" for row in loader.dataset.indices]
    assert labels == set(range(10))
    test = mnist_data.MNISTStream("test")
    assert len(test) == 10000
    assert test[0][2] == "test:0"
    assert not set(ids) & {f"test:{row}" for row in test.indices}


def test_smallresnet_training_update_and_eval_state():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = SmallResNet()
        images, labels, ids = next(iter(mnist_data.get_mnist_loader()))
        model.eval()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            assert model(images).shape == (32, 10)
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = model.classifier[1].weight.detach().clone()
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
        assert not torch.equal(before, model.classifier[1].weight)
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("kwargs", [{"split": "bad"}, {"validation_size": 0},
                                    {"validation_size": 60000}])
def test_invalid_split_settings(kwargs):
    with pytest.raises(ValueError):
        mnist_data.MNISTStream(**kwargs)


def test_required_9000_1000_subset():
    kwargs = {"train_size": 9000, "validation_size": 1000}
    train = mnist_data.get_mnist_loader(**kwargs)
    validation = mnist_data.get_mnist_loader("validation", **kwargs)
    a, b = train.dataset.indices, validation.dataset.indices
    assert len(a) == len(set(a)) == 9000
    assert len(b) == len(set(b)) == 1000
    assert not set(a) & set(b)
    selected = torch.randperm(60000, generator=torch.Generator().manual_seed(42)).tolist()[:10000]
    assert set(a) | set(b) == set(selected)
    assert a == mnist_data.MNISTStream(**kwargs).indices
    other = mnist_data.MNISTStream(stream_seed=43, **kwargs)
    assert a != other.indices and set(a) == set(other.indices)
    assert b == mnist_data.MNISTStream("validation", stream_seed=43, **kwargs).indices
    sizes, ids, labels = [], [], set()
    for images, batch_labels, sample_ids in train:
        assert images.shape == (len(batch_labels), 1, 28, 28)
        sizes.append(len(images)); ids.extend(sample_ids); labels.update(batch_labels.tolist())
    assert sizes == [32] * 281 + [8]
    assert len(train) == 282 and not train.drop_last
    assert len(ids) == len(set(ids)) == 9000
    assert ids == [f"train:{row}" for row in a]
    assert labels == set(range(10))
    assert sum(len(batch[0]) for batch in validation) == 1000


@pytest.mark.parametrize("train_size", [0, -1, 59001, 9000.5, True])
def test_invalid_subset_size(train_size):
    with pytest.raises(ValueError, match="train_size"):
        mnist_data.MNISTStream(train_size=train_size, validation_size=1000)


def test_literal_profile_stratified_membership_and_eval_transform():
    kwargs = dict(profile="nguyen_literal", membership_strategy="nguyen_stratified",
                  train_size=9000, validation_size=1000)
    train = mnist_data.MNISTStream(**kwargs)
    val = mnist_data.MNISTStream("validation", **kwargs)
    assert len(train) == 9000 and len(val) == 1000
    assert not set(train.indices) & set(val.indices)
    assert torch.bincount(train.source.targets[train.indices]).tolist() == [900]*10
    assert torch.bincount(val.source.targets[val.indices]).tolist() == [100]*10
    changed = mnist_data.MNISTStream(stream_seed=43, **kwargs)
    assert set(changed.indices) == set(train.indices) and changed.indices != train.indices
    image, label, sample_id = val[0]
    assert image.shape == (3,32,32) and image.dtype == torch.float32
    assert 0 <= image.min() <= image.max() <= 1
    assert torch.equal(image[0],image[1]) and torch.equal(image[1],image[2])
    assert torch.equal(val[0][0],image)  # evaluation never augments
    assert sample_id == f"train:{val.indices[0]}"
    torch.manual_seed(42); a=train[0][0]
    torch.manual_seed(42); b=train[0][0]
    assert torch.equal(a,b) and a.shape == (3,32,32)


def test_literal_model_has_same_input_head_and_single_normalization():
    from pathlib import Path
    import yaml
    config=yaml.safe_load((Path(__file__).parents[1]/"configs/mnist_nguyen_literal.yaml").read_text())
    assert config["in_channels"] == 3 and config["num_outputs"] == 100
    assert config["eval_batch_size"] == 256 and config["num_workers"] == 2
    assert config["mean"] == [0.5071,0.4865,0.4409]
    assert config["std"] == [0.2673,0.2564,0.2762]
    model=SmallResNet(num_classes=100,in_channels=3,mean=config["mean"],std=config["std"])
    assert model.features[0][0].in_channels == 3
    assert model.classifier[1].out_features == 100
    image = mnist_data.get_mnist_transform("nguyen_literal",False)(Image.new("L",(28,28),255))
    assert image.min() == image.max() == 1  # model alone normalizes
