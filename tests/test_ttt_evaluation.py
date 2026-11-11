import json

import numpy as np
import pytest
import torch

from src.ttt_evaluation import (
    cifar10c_image, evaluate_stream, gaussian_noise, save_manifest,
    score_predictions, validate_split,
)


def test_manifest_preserves_order_and_rejects_overlap(tmp_path):
    path = tmp_path / "split.json"
    manifest = save_manifest(
        path, dataset="example", development_ids=["test:0"],
        evaluation_ids=["test:2", "test:1"], seed=42,
        preprocessing={"scale": "[0,1]"}, corruption={"std": 0.1},
    )
    assert json.loads(path.read_text()) == manifest
    assert manifest["evaluation_ids"] == ["test:2", "test:1"]
    with pytest.raises(ValueError, match="overlap"):
        validate_split(["test:1"], ["test:1"])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_split(["test:0"], ["test:1", "test:1"])


@pytest.mark.parametrize("experiment", ["cifar10c", "cifar100_noise"])
def test_tiny_evaluation_pipeline(experiment, monkeypatch):
    ids = ["test:2", "test:1"]
    clean = torch.full((3, 32, 32), 0.5)
    if experiment == "cifar10c":
        class SyntheticCorruption:
            shape = (50000, 32, 32, 3)
            dtype = np.dtype("uint8")

            def __getitem__(self, index):
                assert index in (20001, 20002)
                return np.full((32, 32, 3), 128, dtype=np.uint8)

        monkeypatch.setattr(np, "load", lambda *args, **kwargs: SyntheticCorruption())
        corrupt = {key: cifar10c_image("fixture.npy", int(key[-1]), 3) for key in ids}
    else:
        corrupt = {key: gaussian_noise(clean, key, seed=42, std=0.1) for key in ids}
        assert torch.equal(corrupt[ids[0]], gaussian_noise(clean, ids[0], seed=42, std=0.1))
        assert not torch.equal(corrupt[ids[0]], corrupt[ids[1]])

    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0.5)
    inputs = []

    def adapt(model, optimizer, image):
        assert image.shape == (3, 32, 32)
        inputs.append((model.weight.item(), len(optimizer.state)))
        optimizer.zero_grad()
        (-model.weight.sum()).backward()
        optimizer.step()

    def predict(model, image):
        assert not model.training and not torch.is_grad_enabled()
        return int(model.weight.item() > 1)

    labels = dict(zip(ids, [0, 1]))
    for images in ({key: clean for key in ids}, corrupt):
        for condition, expected, seen in [
            ("frozen", [0, 0], []),
            ("standard", [0, 0], [(0, 0), (0, 0)]),
            ("online", [0, 1], [(0, 0), (1, 1)]),
            ("online", [0, 1], [(0, 0), (1, 1)]),
        ]:
            inputs.clear()
            rows = evaluate_stream(model, optimizer, images=images, evaluation_ids=ids,
                                   predict=predict, condition=condition, adapt=adapt)
            assert rows == list(zip(ids, expected))
            assert inputs == seen
            assert score_predictions(rows, labels, ids) == (1 if condition == "online" else 0.5)
    assert model.weight.item() == 0 and not optimizer.state
    with pytest.raises(ValueError, match="exactly one"):
        score_predictions([(ids[0], 0), (ids[0], 0)], labels, ids)


def test_frozen_rejects_state_mutation():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1)

    def bad_predict(model, image):
        model.weight.add_(1)
        return 0

    with pytest.raises(ValueError, match="changed"):
        evaluate_stream(model, optimizer, images={"test:0": torch.zeros(1)},
                        evaluation_ids=["test:0"], predict=bad_predict, condition="frozen")
