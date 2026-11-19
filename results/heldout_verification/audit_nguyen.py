"""Audit the saved notebooks and test the actual runner with fixture weights."""
import argparse
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import random
import subprocess
import time
import math
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

NOTEBOOK = "notebooks/nguyen-expa.ipynb"
REVISIONS = ("f2aa445", "64ae8d1")


def source(notebook, index):
    return "".join(notebook["cells"][index]["source"])


def config(notebook, *keys):
    tree = ast.parse(source(notebook, 2))
    table = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "CFG" for t in node.targets))
    return {key.value: ast.literal_eval(value) for key, value in zip(table.keys, table.values)
            if isinstance(key, ast.Constant) and key.value in keys}


def split(settings):
    rng = np.random.RandomState(settings["seed"])
    train = rng.permutation(50_000)
    test = rng.permutation(10_000)
    source_dev_size = settings["source_dev_size"]
    dev_size = settings["ttt_dev_size"]
    adaptation = np.sort(test[dev_size:])
    stream_order = np.random.RandomState(settings["seed"] + 1).permutation(len(adaptation))
    return {
        "source_train_ids": [f"train:{i}" for i in sorted(train[source_dev_size:])],
        "source_development_ids": [f"train:{i}" for i in sorted(train[:source_dev_size])],
        "adaptation_ids_in_order": [f"test:{i}" for i in adaptation[stream_order]],
        "held_out_ids_in_order": [f"test:{i}" for i in sorted(test[:dev_size])],
    }


def outputs(notebook, index):
    for output in notebook["cells"][index].get("outputs", []):
        value = output.get("text") or output.get("data", {}).get("text/plain", [])
        yield "".join(value)


def equal(left, right):
    if isinstance(left, (torch.Tensor, np.ndarray)):
        return torch.equal(left, right) if isinstance(left, torch.Tensor) else np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def check_runner(notebook, settings, ids):
    # Execute only reviewed function definitions. No data downloads, source
    # training, or notebook experiment cells run here.
    def seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    scope = dict(torch=torch, F=F, np=np, copy=copy, time=time, math=math,
                 CFG=settings, NOTES=[], SEEDS={"ttt_augmentation": settings["seed"] + 100},
                 set_seed=seed_all, to_float=lambda x: x)
    names = {"predict", "run_ttt", "make_monitor", "frozen_snapshot", "frozen_unchanged"}
    for index in (8, 12, 14):
        tree = ast.parse(source(notebook, index))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(tree, f"reviewed-cell-{index}", "exec"), scope)

    # Encode the original row number in a feature to trace each prediction.
    # The small model and synthetic augmentation exercise control flow only.
    def batch(x):
        varied = x.clone()
        varied[:, 1] += torch.rand(len(x)) + np.random.random() + random.random()
        return varied, torch.arange(len(x)) % 4

    scope["make_ttt_batch"] = batch
    scope["make_chunk_batch"] = batch
    active_trace = None

    class FixtureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Linear(2, 4)
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(4, 10))
            self.rotation = nn.Linear(4, 4)
            self.register_buffer("scale", torch.tensor([10_000.0, 1.0]))

        def classify(self, x):
            if active_trace is not None:
                active_trace.extend(f"test:{int(i)}" for i in x[:, 0].tolist())
            return self.head(self.shared(x / self.scale))

        def rotate_logits(self, x):
            return self.rotation(self.shared(x / self.scale))

    optimizers = []

    class TracedSGD(torch.optim.SGD):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.steps_seen = 0
            optimizers.append(self)

        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            self.steps_seen += 1
            return result

    def fixture(rows, noisy):
        return torch.tensor([[int(row.split(":")[1]), float(noisy)] for row in rows])

    held = ids["held_out_ids_in_order"]
    scope["TDEV"] = {
        key: (fixture(held, noisy), np.arange(len(held)) % 10)
        for key, noisy in (("clean", False), ("noisy", True))
    }
    seed_all(0)
    base = FixtureModel().eval()
    start = copy.deepcopy(base.state_dict())
    interval = settings["monitor_every"]
    expected = list(range(0, 8001, interval))
    assert "y" not in inspect.signature(scope["run_ttt"]).parameters
    for stream, noisy in (("clean", False), ("noisy", True)):
        records = []
        checkpoints = []
        score = scope["make_monitor"](records, stream)
        opt_count = len(optimizers)

        def monitor(model, count):
            nonlocal active_trace
            opt = optimizers[opt_count]
            assert len(optimizers) == opt_count + 1
            assert opt.steps_seen == count
            assert not model.training
            assert equal(model.state_dict(), start) if count == 0 else not equal(model.state_dict(), start)
            snapshot = lambda: (copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()),
                                torch.get_rng_state(), random.getstate(), np.random.get_state())
            before = snapshot()
            active_trace = []
            score(model, count)
            assert active_trace == held + held
            active_trace = None
            assert equal(before, snapshot())
            assert [(row["heldout_set"], row["heldout_n"]) for row in records[-2:]] == [
                ("clean", len(held)), ("noisy", len(held))]
            checkpoints.append(count)

        with patch.object(torch.optim, "SGD", TracedSGD):
            pred, info = scope["run_ttt"](
                base, fixture(ids["adaptation_ids_in_order"], noisy), "online",
                settings["online_lr"], settings["online_steps"], log_every=0,
                update_images=settings["online_update_images"], monitor=monitor, monitor_every=interval,
            )
        assert checkpoints == expected
        assert len(records) == 162 and len(pred) == 8000 and info["heads_frozen_ok"]
        assert equal(base.state_dict(), start)
        print(f"FIXTURE PASS {stream}: {len(checkpoints)} checkpoints, {len(records)} scores, "
              f"{len(records) * len(held)} ordered held-out predictions, "
              "one optimizer and 8,000 updates; state and RNG unchanged during scoring")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaggle-export", type=Path, required=True)
    args = parser.parse_args()
    raw = [subprocess.check_output(["git", "show", f"{rev}:{NOTEBOOK}"]) for rev in REVISIONS]
    old, current = map(json.loads, raw)
    export = json.loads(args.kaggle_export.read_bytes())
    kaggle = json.loads(export["blob"]["source"])
    assert len(current["cells"]) == len(kaggle["cells"])
    code_cells = [i for i, cell in enumerate(current["cells"]) if cell["cell_type"] == "code"]
    assert all(source(current, i) == source(kaggle, i) for i in code_cells)
    print(f"Saved GitHub notebook {REVISIONS[1]} SHA256 {hashlib.sha256(raw[1]).hexdigest()}")
    print(f"Kaggle version {export['metadata']['currentVersionNumber']}: all {len(code_cells)} code cells match GitHub")
    keys = ("seed", "source_dev_size", "ttt_dev_size", "quick", "dev_limit", "eval_limit")
    old_cfg = config(old, *keys, "tune_ttt", "tune_episodic_n")
    settings = config(current, *keys, "monitor_every", "online_lr", "online_steps",
                      "episodic_lr", "episodic_steps", "online_update_images", "ttt_momentum",
                      "ttt_batch", "corruption", "severity", "gn_groups", "ttt_augment")
    assert not settings["quick"] and settings["dev_limit"] is None and settings["eval_limit"] is None
    ids = split(settings)
    combined = sum(ids.values(), [])
    assert list(map(len, ids.values())) == [45000, 5000, 8000, 2000]
    assert len(combined) == len(set(combined)) == 60000
    prior = split(old_cfg)["held_out_ids_in_order"]
    assert prior == ids["held_out_ids_in_order"] and old_cfg["tune_ttt"]
    print("PASS target membership: 45,000 source train, 5,000 source dev, 8,000 adaptation, 2,000 quiz; zero duplicate source IDs")
    print("FAIL independence: 2,000/2,000 quiz IDs match the earlier online tuning pool")
    print("Earlier tuned settings saved by the notebook:")
    for result in outputs(old, 14):
        for line in result.splitlines():
            if line.startswith("chosen TTT settings:"):
                print(line)
    print("Latest fixed settings:", json.dumps(settings, sort_keys=True))
    print("Saved source training log (final epoch):")
    for result in outputs(current, 10):
        for line in result.splitlines():
            if "ep 30/30" in line:
                print(line)
    results = "\n".join(outputs(current, 16))
    monitor_output = results.split("held-out ttt_dev accuracy during online TTT", 1)[1]
    monitor_table = [line for line in monitor_output.splitlines()
                     if line.strip().startswith(("stream ", "heldout_set ", "images_adapted"))
                     or line.split()[:1] in (["0"], ["4000"], ["8000"]) and len(line.split()) == 5]
    assert any(line.split()[:1] == ["8000"] for line in monitor_table)
    print("Reported trained monitor excerpt (not an independent replay):")
    print("\n".join(monitor_table))
    print("Reported noisy adaptation-stream score (separate from the fixed monitor):")
    for line in results.splitlines():
        if "0.724500" in line:
            print(line)
    print("UNVERIFIED trained schedule at every point, per-ID score logs, and state continuity: raw run artifacts unavailable")
    destination = Path(__file__).resolve().parent / "nguyen-original-split.json"
    manifest = {"reconstructed_from": REVISIONS[1], "settings": settings,
                "adaptation_order_seed": settings["seed"] + 1, "augmentation_seed": settings["seed"] + 100,
                "prior_ttt_development_overlap": len(prior), **ids}
    destination.write_text(json.dumps(manifest, separators=(",", ":")) + "\n", encoding="utf-8")
    torch.set_num_threads(2)
    check_runner(current, settings, ids)


if __name__ == "__main__":
    main()
