"""Exercise the reviewed Kaggle notebook definitions without executing training cells."""
import argparse
import ast
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10

from src.ttt_evaluation import cifar10c_image

REVIEWED_SOURCE_SHA256 = "72ecb10b08a32a2ad0add11a9905f75ad6d28afb0a639e0a9519f2b2805c7197"


def load_reviewed_definitions(path):
    export = json.loads(path.read_bytes())
    source = export["blob"]["source"] if "blob" in export else path.read_text(encoding="utf-8")
    if hashlib.sha256(source.encode()).hexdigest() != REVIEWED_SOURCE_SHA256:
        raise ValueError("Notebook differs from the reviewed version; review it before running this check")
    notebook = json.loads(source)
    scope = dict(np=np, torch=torch, nn=nn, F=F, copy=copy, time=time, math=math,
                 CFG=dict(gn_groups=8, ttt_batch=32, ttt_augment=True, ttt_momentum=0),
                 NOTES=[], SEEDS={"ttt_augmentation": 100})
    scope["to_float"] = lambda x: x.float().div(255)

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    scope["set_seed"] = set_seed
    wanted = {"Normalize", "conv_gn_relu", "Residual", "stage", "TTTNet", "augment",
              "rotate_batch", "predict", "make_ttt_batch", "frozen_snapshot",
              "frozen_unchanged", "run_ttt"}
    for cell in (6, 8, 12):
        tree = ast.parse("".join(notebook["cells"][cell]["source"]))
        tree.body = [
            node for node in tree.body
            if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted)
            or (isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in {"CIFAR_MEAN", "CIFAR_STD"}
                for target in node.targets))
        ]
        exec(compile(tree, f"reviewed-notebook-cell-{cell}", "exec"), scope)
    return scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--checkpoint", type=Path, help="Optional joint_best.pt from Nguyen")
    args = parser.parse_args()
    scope = load_reviewed_definitions(args.notebook)
    torch.set_num_threads(2)
    dataset = CIFAR10(args.root, train=False, download=False)
    rng = np.random.RandomState(0)
    source_order = rng.permutation(50000)
    test_order = rng.permutation(10000)
    assert not np.intersect1d(source_order[:5000], source_order[5000:]).size
    development, evaluation = np.sort(test_order[:2000]), np.sort(test_order[2000:])
    assert not np.intersect1d(development, evaluation).size
    corruption = args.root / "CIFAR-10-C"
    labels = np.load(corruption / "labels.npy", allow_pickle=False)
    assert np.array_equal(labels[-10000:], dataset.targets)
    selected = evaluation[:2]
    clean = torch.from_numpy(dataset.data[selected]).permute(0, 3, 1, 2).contiguous()
    noisy = torch.stack([
        (cifar10c_image(corruption / "gaussian_noise.npy", int(row), 5) * 255).round().byte()
        for row in selected
    ])

    scope["set_seed"](0)
    model = scope["TTTNet"]().eval()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    starts = []

    # deepcopy copies hooks. Capture the shared extractor before each adaptation
    # batch (batch 32), excluding classification forwards (batch 1).
    def observe(module, inputs):
        if inputs[0].shape[0] == 32:
            starts.append(tuple(p.detach().clone() for p in module.parameters()))

    hook = model.shared.register_forward_pre_hook(observe)
    print("Model:", "joint checkpoint" if args.checkpoint else "random initialization")
    print("PASS: severity-5 labels align; 2000 development / 8000 eval, zero overlap")
    print("Smoke source rows:", selected.tolist())
    for name, images in (("clean", clean), ("severity-5", noisy)):
        frozen = scope["predict"](model, images)
        assert len(frozen) == len(images)
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in initial.items())
        for mode in ("episodic", "online"):
            starts.clear()
            prediction, info = scope["run_ttt"](model, images, mode, 0.001, 1, log_every=0)
            assert len(prediction) == len(images) and np.all((prediction >= 0) & (prediction < 10))
            assert len(starts) == 2 and info["heads_frozen_ok"] and info["final_rel_param_drift"] > 0
            same = all(torch.equal(a, b) for a, b in zip(*starts))
            assert same == (mode == "episodic")
            first = starts[0]
            starts.clear()
            repeat, _ = scope["run_ttt"](model, images, mode, 0.001, 1, log_every=0)
            assert np.array_equal(prediction, repeat)
            assert all(torch.equal(a, b) for a, b in zip(first, starts[0]))
            assert all(torch.equal(value, model.state_dict()[key]) for key, value in initial.items())
            print(f"PASS: {name} {mode}: two predictions, expected state behavior, independent rerun")
    hook.remove()
    try:
        bad, _ = scope["run_ttt"](model, noisy, "online", 0.001, 1, order=[0, 0], log_every=0)
    except ValueError:
        print("PASS: duplicate order rejected")
    else:
        if np.any(bad == -1):
            print("FAIL: duplicate order accepted; at least one prediction missing")
    print("REVIEW: run_ttt accepts class labels for progress logging; move scoring outside it")


if __name__ == "__main__":
    main()
