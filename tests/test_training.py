import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.train_online import classification_metrics, evaluate, train_one_pass


def test_metrics_known_confusion():
    metrics = classification_metrics(torch.tensor([[2, 1], [0, 1]]))
    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 0.75
    assert metrics["recall"] == pytest.approx(5 / 6)
    assert metrics["f1"] == pytest.approx((0.8 + 2 / 3) / 2)


def test_one_pass_and_evaluation_preserve_state():
    torch.manual_seed(42)
    source = TensorDataset(torch.randn(70, 3), torch.arange(70) % 2, torch.arange(70))
    loader = DataLoader(source, batch_size=32, shuffle=False)
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 2))
    model.train()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    evaluate(model, loader, "cpu", 2)
    assert model.training
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    checkpoints = []
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    examples, updates, seconds = train_one_pass(
        model, loader, optimizer, "cpu",
        lambda n, u, s: checkpoints.append((n, u)), 64)
    assert examples == 70 and updates == 3 and seconds > 0
    assert checkpoints == [(0, 0), (64, 2), (70, 3)]
    assert not torch.equal(before["1.weight"], model.state_dict()["1.weight"])


def test_final_checkpoint_not_duplicated():
    source = TensorDataset(torch.randn(70, 3), torch.arange(70) % 2, torch.arange(70))
    model = torch.nn.Linear(3, 2)
    checkpoints = []
    train_one_pass(model, DataLoader(source, batch_size=32),
                   torch.optim.SGD(model.parameters(), lr=0.01), "cpu",
                   lambda n, u, s: checkpoints.append(n), 65)
    assert checkpoints == [0, 70]
