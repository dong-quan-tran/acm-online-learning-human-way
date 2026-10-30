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


def test_nguyen_adamw_schedule_matches_direct_reference():
    """Compare all 282 update rates/beta1 values with Nguyen's constructor."""
    from pathlib import Path
    import yaml
    from src.train_online import build_optimizer_and_scheduler
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/mnist_nguyen_optimizer.yaml").read_text())
    assert config["optimizer"] == "adamw"
    assert config["learning_rate"] == config["weight_decay"] == 5e-4
    assert config["dropout"] == .2
    assert config["model_seed"] == config["split_seed"] == config["stream_seed"] == 42
    model, reference_model = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config, 282)
    reference = torch.optim.AdamW(reference_model.parameters(), lr=5e-4, weight_decay=5e-4)
    reference_schedule = torch.optim.lr_scheduler.OneCycleLR(
        reference, max_lr=5e-4, total_steps=282, pct_start=.15)
    assert isinstance(optimizer, torch.optim.AdamW)
    for _ in range(282):
        assert optimizer.param_groups[0]["lr"] == reference.param_groups[0]["lr"]
        assert optimizer.param_groups[0]["betas"] == reference.param_groups[0]["betas"]
        assert optimizer.param_groups[0]["weight_decay"] == 5e-4
        optimizer.step(); scheduler.step()
        reference.step(); reference_schedule.step()
    assert scheduler.last_epoch == reference_schedule.last_epoch == 282


def test_scheduler_steps_once_per_training_update():
    from src.train_online import build_optimizer_and_scheduler
    model = torch.nn.Linear(3, 2)
    loader = DataLoader(TensorDataset(torch.randn(9000, 3), torch.arange(9000) % 2,
                                     torch.arange(9000)), batch_size=32, shuffle=False)
    config = {"optimizer": "adamw", "learning_rate": 5e-4, "weight_decay": 5e-4,
              "scheduler": "OneCycleLR", "pct_start": .15}
    optimizer, scheduler = build_optimizer_and_scheduler(model, config, len(loader))
    seen = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        examples, updates, _ = train_one_pass(model, loader, optimizer, "cpu",
            lambda n,u,s: seen.append((n,u,scheduler.last_epoch)), 1000, scheduler=scheduler)
    finally:
        torch.set_num_threads(previous_threads)
    assert examples == 9000 and updates == scheduler.last_epoch == 282
    assert seen[-1] == (9000,282,282)
    assert all(updates == steps for _, updates, steps in seen)


def test_historical_adam_settings_preserved():
    from src.train_online import build_optimizer_and_scheduler
    model = torch.nn.Linear(3, 2)
    optimizer, scheduler = build_optimizer_and_scheduler(model,
        {"optimizer": "adam", "learning_rate": .001}, 282)
    assert type(optimizer) is torch.optim.Adam
    assert optimizer.param_groups[0]["lr"] == .001
    assert optimizer.param_groups[0]["weight_decay"] == 0
    assert scheduler is None


def test_hundred_output_metrics_include_outside_predictions():
    matrix=torch.zeros(100,100,dtype=torch.int64)
    for digit in range(10): matrix[digit,digit]=9
    matrix[0,99]=10
    metrics=classification_metrics(matrix,label_classes=range(10))
    assert metrics["accuracy"] == .9
    assert metrics["recall"] == pytest.approx((9/19+9)/10)
    assert metrics["precision"] == 1.0
