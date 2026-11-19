"""Verification suite for Experiment B periodic held-out evaluation (docs/11.14-11.20_assignment.md)."""

from __future__ import annotations

import copy
import csv
import inspect
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Dict, List

import numpy as np
from PIL import Image
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src import data as data_mod
from src.data import (
    CIFAR100SplitSubsetDataset,
    get_exp_b_split_loaders,
    load_exp_b_split_ids,
    load_exp_b_split_pair,
    parse_source_id,
)
from src.models import SmallResNetTTT
from src.ttt_held_out import (
    HELD_OUT_CSV_COLUMNS,
    _assert_rng_state_unchanged,
    _capture_rng_state,
    adapt_image_only,
    evaluate_held_out_set,
    prepare_held_out_batches,
    run_held_out_experiment,
    run_held_out_stream,
    save_held_out_csv,
)


class SyntheticCIFAR:
    """Offline synthetic CIFAR-like RGB dataset returning fine_label in 0..19 (and configurable outliers)."""

    def __init__(self, num_samples: int = 1000, *, bad_label_rows: Dict[int, int] | None = None):
        self.num_samples = num_samples
        self.bad_label_rows = bad_label_rows or {}

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        lbl = self.bad_label_rows.get(idx, idx % 20)
        return {
            "img": Image.new(
                "RGB",
                (32, 32),
                color=(idx % 256, (idx * 7) % 256, (idx * 13) % 256),
            ),
            "fine_label": lbl,
        }


def _make_loader(
    refs: List[tuple[str, int]],
    *,
    corruption: str = "clean",
    batch_size: int = 1,
    num_samples: int = 1000,
) -> DataLoader:
    ds = CIFAR100SplitSubsetDataset(
        refs,
        corruption=corruption,
        noise_sigma=0.08,
        noise_seed=42,
        datasets_by_split={"test": SyntheticCIFAR(num_samples=num_samples)},
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)


# ==============================================================================
# 1. Split Parsing, Order Preservation & Disjointness
# ==============================================================================


class TestSplitLoadingAndValidation:
    def test_parse_source_id_accepts_int_and_prefixed_strings(self):
        assert parse_source_id(0) == ("test", 0)
        assert parse_source_id(42, default_source_split="train") == ("train", 42)
        assert parse_source_id("test:17") == ("test", 17)
        assert parse_source_id("train:512") == ("train", 512)

        for bad in (True, False, -1, "17", "val:10", "test:-5", "test:abc", 1.5, None):
            with pytest.raises(ValueError):
                parse_source_id(bad)  # type: ignore[arg-type]

    def test_load_combined_and_separate_splits_preserve_order(self, tmp_path: Path):
        adapt_order = ["test:9", "test:2", "test:5"]
        held_order = ["test:20", "test:11"]
        combined_path = tmp_path / "split.json"
        combined_path.write_text(
            json.dumps({"adaptation_ids": adapt_order, "held_out_ids": held_order}),
            encoding="utf-8",
        )
        adapt_refs, held_refs, meta = load_exp_b_split_pair(
            combined_path, train_val_membership=None, prior_test_membership=None
        )
        assert adapt_refs == [("test", 9), ("test", 2), ("test", 5)]
        assert held_refs == [("test", 20), ("test", 11)]
        assert meta["adaptation_count"] == 3
        assert meta["held_out_count"] == 2

        # Separate files using integer IDs + fallback 'indices'
        f_adapt = tmp_path / "adapt.json"
        f_held = tmp_path / "held.json"
        f_adapt.write_text(json.dumps({"source_split": "test", "indices": [9, 2, 5]}))
        f_held.write_text(json.dumps({"source_split": "test", "indices": [20, 11]}))
        a2, h2, _ = load_exp_b_split_pair(
            f_adapt, f_held, train_val_membership=None, prior_test_membership=None
        )
        assert a2 == adapt_refs
        assert h2 == held_refs

    def test_rejects_overlap_duplicates_and_source_leakage(self, tmp_path: Path):
        # 1. Overlap between adaptation and held-out
        overlap_path = tmp_path / "overlap.json"
        overlap_path.write_text(
            json.dumps({"adaptation": [1, 2, 3], "held_out": [3, 4]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="overlap"):
            load_exp_b_split_pair(
                overlap_path, train_val_membership=None, prior_test_membership=None
            )

        # 2. Duplicate inside split
        dup_path = tmp_path / "dup.json"
        dup_path.write_text(
            json.dumps({"adaptation": [1, 1, 2], "held_out": [3, 4]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_exp_b_split_pair(
                dup_path, train_val_membership=None, prior_test_membership=None
            )

        # 3. Overlap with train/validation membership
        tv_path = tmp_path / "train_val.json"
        tv_path.write_text(
            json.dumps({"source_split": "train", "train": [10, 11], "validation": [12]})
        )
        leak_path = tmp_path / "leak.json"
        leak_path.write_text(
            json.dumps({"adaptation_ids": ["train:12"], "held_out_ids": ["test:0"]})
        )
        with pytest.raises(ValueError, match="train/validation"):
            load_exp_b_split_pair(
                leak_path, train_val_membership=tv_path, prior_test_membership=None
            )

    def test_tracks_prior_exp_b_overlap_metadata_and_rejects_classes_over_19(self, tmp_path: Path):
        prior_path = tmp_path / "prior.json"
        prior_path.write_text(json.dumps({"source_split": "test", "test": [100, 101]}))
        split_path = tmp_path / "split.json"
        split_path.write_text(
            json.dumps({"adaptation_ids": ["test:200"], "held_out_ids": ["test:100", "test:300"]})
        )
        _, _, meta = load_exp_b_split_pair(
            split_path, train_val_membership=None, prior_test_membership=prior_path
        )
        assert meta["held_out_examined_in_prior_exp_b"] is True
        assert meta["held_out_prior_overlap_count"] == 1

        # Reject class label >= 20
        bad_synth = SyntheticCIFAR(num_samples=50, bad_label_rows={5: 20})
        with pytest.raises(ValueError, match="outside allowed classes"):
            CIFAR100SplitSubsetDataset(
                [("test", 0), ("test", 5)],
                datasets_by_split={"test": bad_synth},
            )

    def test_clean_and_noisy_held_out_aligned_and_deterministic(self, tmp_path: Path):
        split_path = tmp_path / "split.json"
        split_path.write_text(
            json.dumps({
                "adaptation_ids": [f"test:{i}" for i in range(10)],
                "held_out_ids": [f"test:{i}" for i in range(20, 30)],
            })
        )
        synth = SyntheticCIFAR(num_samples=100)
        _, ho_clean, ho_noisy, _ = get_exp_b_split_loaders(
            split_path,
            datasets_by_split={"test": synth},
            train_val_membership=None,
            prior_test_membership=None,
            eval_batch_size=4,
        )
        clean_imgs, clean_lbls, clean_ids = [], [], []
        for imgs, lbls, sids in ho_clean:
            clean_imgs.append(imgs)
            clean_lbls.extend(lbls.tolist())
            clean_ids.extend(sids)
        noisy_imgs, noisy_lbls, noisy_ids = [], [], []
        for imgs, lbls, sids in ho_noisy:
            noisy_imgs.append(imgs)
            noisy_lbls.extend(lbls.tolist())
            noisy_ids.extend(sids)

        assert clean_ids == noisy_ids == [f"test:{i}" for i in range(20, 30)]
        assert clean_lbls == noisy_lbls
        c_cat = torch.cat(clean_imgs, dim=0)
        n_cat = torch.cat(noisy_imgs, dim=0)
        assert not torch.equal(c_cat, n_cat)
        assert n_cat.min() >= 0.0 and n_cat.max() <= 1.0


# ==============================================================================
# 2. Label-Free Contract & State Invariance During Held-Out Eval
# ==============================================================================


class TestInvarianceAndContract:
    def test_adapt_image_only_strict_label_free_signature(self):
        sig = inspect.signature(adapt_image_only)
        params = list(sig.parameters.values())
        names = [p.name for p in params]
        assert not any(
            p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for p in params
        )
        forbidden = ("label", "target", "gt", "ground_truth", "class", "y")
        for name in names:
            for bad in forbidden:
                assert bad not in name.lower(), f"Forbidden token {bad!r} in param {name!r}"

    def test_adapt_image_only_rejects_head_optimizer_and_updates_features_only(self):
        model = SmallResNetTTT(num_classes=100)
        bad_opt = torch.optim.SGD(model.parameters(), lr=0.01)
        with pytest.raises(ValueError, match="features"):
            adapt_image_only(model, torch.rand(1, 3, 32, 32), optimizer=bad_opt)

        good_opt = torch.optim.SGD(model.features.parameters(), lr=0.01, momentum=0.9)
        cls_before = {k: v.clone() for k, v in model.classifier.state_dict().items()}
        rot_before = {k: v.clone() for k, v in model.rotation_head.state_dict().items()}
        feat_before = {k: v.clone() for k, v in model.features.state_dict().items()}

        model.train()
        loss = adapt_image_only(model, torch.rand(1, 3, 32, 32), optimizer=good_opt)
        assert loss > 0.0
        assert not model.training
        for k in cls_before:
            assert torch.equal(model.classifier.state_dict()[k], cls_before[k])
        for k in rot_before:
            assert torch.equal(model.rotation_head.state_dict()[k], rot_before[k])
        assert any(
            not torch.equal(model.features.state_dict()[k], feat_before[k])
            for k in feat_before
        )
        assert len(good_opt.state) > 0  # Stateful momentum buffers populated

    def test_held_out_eval_preserves_model_optimizer_and_rng_states(self):
        model = SmallResNetTTT(num_classes=100)
        opt = torch.optim.SGD(model.features.parameters(), lr=0.001, momentum=0.9)
        # Step once so optimizer holds non-empty momentum buffers
        adapt_image_only(model, torch.rand(1, 3, 32, 32), optimizer=opt, ttt_lr=0.001)

        ho_loader = _make_loader([("test", i) for i in range(25)], batch_size=8)
        batches, expected_ids = prepare_held_out_batches(ho_loader, batch_size=8)

        # Seed all RNGs deterministically and record next draws AFTER held-out eval vs reference
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        expected_py = random.random()
        expected_np = float(np.random.rand())
        expected_th = float(torch.rand(1).item())

        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        res = evaluate_held_out_set(
            model, batches, expected_ids, device="cpu", optimizer=opt, verify_invariants=True
        )
        assert res.image_count == 25
        assert len(res.predictions) == 25
        assert random.random() == expected_py
        assert float(np.random.rand()) == expected_np
        assert float(torch.rand(1).item()) == expected_th

    def test_batched_held_out_eval_matches_batch_size_1_predictions(self):
        ckpt_path = Path("checkpoints/exp_b_joint_seed42.pt")
        if not ckpt_path.is_file():
            pytest.skip("Seed-42 joint checkpoint not present")
        from src.ttt_cifar100 import load_model

        model, _ = load_model(ckpt_path, device="cpu")
        refs = [("test", i) for i in range(30)]
        loader = _make_loader(refs, corruption="gaussian_noise", batch_size=1)
        b1, ids1 = prepare_held_out_batches(loader, batch_size=1)
        b64, ids64 = prepare_held_out_batches(loader, batch_size=64)

        r1 = evaluate_held_out_set(model, b1, ids1, device="cpu")
        r64 = evaluate_held_out_set(model, b64, ids64, device="cpu")
        assert r1.predictions == r64.predictions
        assert r1.correct_count == r64.correct_count

    def test_held_out_eval_detects_state_or_rng_corruption(self):
        class MutatingModel(SmallResNetTTT):
            def forward(self, x):
                torch.rand(1)  # Corrupts global PyTorch RNG!
                return super().forward(x)

        model = MutatingModel(num_classes=100)
        ho_loader = _make_loader([("test", 0), ("test", 1)], batch_size=2)
        batches, expected_ids = prepare_held_out_batches(ho_loader, batch_size=2)
        with pytest.raises(RuntimeError, match="random-number state"):
            evaluate_held_out_set(model, batches, expected_ids, device="cpu", verify_invariants=True)


# ==============================================================================
# 3. Periodic Schedule, Carryover Exactness & Split Timing
# ==============================================================================


class TestScheduleAndCarryover:
    @pytest.mark.parametrize(
        "n_adapt,interval,expected_points",
        [
            (200, 100, [0, 100, 200]),
            (237, 100, [0, 100, 200, 237]),
            (45, 100, [0, 45]),
        ],
    )
    def test_checkpoint_schedule_handles_multiples_and_remainders(
        self, n_adapt: int, interval: int, expected_points: List[int]
    ):
        torch.manual_seed(0)
        model = SmallResNetTTT(num_classes=100)
        adapt_loader = _make_loader([("test", i) for i in range(n_adapt)], batch_size=1)
        ho_clean, ids_clean = prepare_held_out_batches(
            _make_loader([("test", 500 + i) for i in range(16)], batch_size=8), batch_size=8
        )
        ho_noisy, ids_noisy = prepare_held_out_batches(
            _make_loader(
                [("test", 500 + i) for i in range(16)], corruption="gaussian_noise", batch_size=8
            ),
            batch_size=8,
        )

        rows, audit = run_held_out_stream(
            model=model,
            adaptation_loader=adapt_loader,
            held_out_clean_batches=ho_clean,
            held_out_clean_ids=ids_clean,
            held_out_noisy_batches=ho_noisy,
            held_out_noisy_ids=ids_noisy,
            condition="online_ttt",
            adaptation_stream_condition="clean",
            eval_interval=interval,
            ttt_lr=0.001,
            ttt_steps=1,
            device="cpu",
        )
        assert audit["checkpoints_evaluated"] == expected_points
        assert len(rows) == 2 * len(expected_points)  # clean + noisy per checkpoint
        recorded = sorted({r["adaptation_images_processed"] for r in rows})
        assert recorded == expected_points
        # At step 0, adapt time & update count are 0; final step reflects all images
        step0_rows = [r for r in rows if r["adaptation_images_processed"] == 0]
        assert all(r["adaptation_time_seconds"] == 0.0 and r["adaptation_update_count"] == 0 for r in step0_rows)
        final_rows = [r for r in rows if r["adaptation_images_processed"] == n_adapt]
        assert all(r["adaptation_update_count"] == n_adapt and r["adaptation_time_seconds"] > 0.0 for r in final_rows)

    def test_evaluation_pauses_do_not_alter_subsequent_adaptation_updates(self):
        """Pausing at 100 images must leave adapted weights identical to uninterrupted adaptation."""
        torch.manual_seed(7)
        base_model = SmallResNetTTT(num_classes=100)
        model_paused = copy.deepcopy(base_model)
        model_unpaused = copy.deepcopy(base_model)

        adapt_refs = [("test", i) for i in range(200)]
        adapt_loader = _make_loader(adapt_refs, corruption="gaussian_noise", batch_size=1)
        ho_clean, ids_clean = prepare_held_out_batches(
            _make_loader([("test", 600 + i) for i in range(12)], batch_size=6), batch_size=6
        )
        ho_noisy, ids_noisy = prepare_held_out_batches(
            _make_loader(
                [("test", 600 + i) for i in range(12)], corruption="gaussian_noise", batch_size=6
            ),
            batch_size=6,
        )

        # Run 1: pause at 100 and 200 (with momentum=0.9 to stress stateful optimizer continuity)
        run_held_out_stream(
            model=model_paused,
            adaptation_loader=adapt_loader,
            held_out_clean_batches=ho_clean,
            held_out_clean_ids=ids_clean,
            held_out_noisy_batches=ho_noisy,
            held_out_noisy_ids=ids_noisy,
            condition="online_ttt",
            adaptation_stream_condition="noisy",
            eval_interval=100,
            ttt_lr=0.001,
            ttt_momentum=0.9,
            device="cpu",
        )
        # Run 2: no intermediate pause (interval=200 -> only [0, 200])
        run_held_out_stream(
            model=model_unpaused,
            adaptation_loader=adapt_loader,
            held_out_clean_batches=ho_clean,
            held_out_clean_ids=ids_clean,
            held_out_noisy_batches=ho_noisy,
            held_out_noisy_ids=ids_noisy,
            condition="online_ttt",
            adaptation_stream_condition="noisy",
            eval_interval=200,
            ttt_lr=0.001,
            ttt_momentum=0.9,
            device="cpu",
        )

        for k, v in model_paused.state_dict().items():
            assert torch.equal(v, model_unpaused.state_dict()[k]), (
                f"Intermediate held-out eval at 100 altered final parameter '{k}'"
            )


# ==============================================================================
# 4. End-to-End Runner, Protected Legacy Outputs & CLI Integration
# ==============================================================================


class TestEndToEndAndCLI:
    def test_save_held_out_csv_protects_legacy_files_and_supports_schema_map(self, tmp_path: Path):
        sample_row = {col: 0 for col in HELD_OUT_CSV_COLUMNS}
        sample_row.update({
            "experiment": "B",
            "condition": "joint_frozen",
            "adaptation_stream_condition": "none",
            "held_out_condition": "clean",
            "held_out_image_count": 10,
            "correct_prediction_count": 5,
            "held_out_accuracy": 0.5,
        })

        with pytest.raises(ValueError, match="Refusing to overwrite legacy"):
            save_held_out_csv([sample_row], tmp_path / "ttt_summary.csv")
        with pytest.raises(ValueError, match="Refusing to overwrite legacy"):
            save_held_out_csv([sample_row], tmp_path / "ttt_window_summary.csv")

        out_path = save_held_out_csv(
            [sample_row],
            tmp_path / "custom.csv",
            schema_map={"held_out_accuracy": "eval_acc", "adaptation_images_processed": "step"},
        )
        with open(out_path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            assert "eval_acc" in reader.fieldnames
            assert "step" in reader.fieldnames
            assert float(rows[0]["eval_acc"]) == 0.5

    def test_end_to_end_experiment_fresh_checkpoints_and_csv_fields(self, tmp_path: Path):
        torch.manual_seed(42)
        sup_model = SmallResNetTTT(num_classes=100)
        joint_model = SmallResNetTTT(num_classes=100)
        sup_ckpt = tmp_path / "sup.pt"
        joint_ckpt = tmp_path / "joint.pt"
        torch.save({"model_state_dict": sup_model.state_dict()}, sup_ckpt)
        torch.save({"model_state_dict": joint_model.state_dict()}, joint_ckpt)

        split_path = tmp_path / "split.json"
        split_path.write_text(
            json.dumps({
                "adaptation_ids": [f"test:{i}" for i in range(230)],
                "held_out_ids": [f"test:{i}" for i in range(500, 520)],
            })
        )
        out_csv = tmp_path / "exp_b_held_out.csv"

        rows, report = run_held_out_experiment(
            split_file=split_path,
            output_csv=out_csv,
            supervised_ckpt=sup_ckpt,
            joint_ckpt=joint_ckpt,
            eval_interval=100,
            eval_batch_size=10,
            datasets_by_split={"test": SyntheticCIFAR(num_samples=600)},
            train_val_membership=None,
            prior_test_membership=None,
            device="cpu",
        )
        assert report["all_checks_passed"] is True
        assert out_csv.is_file()
        assert (tmp_path / "exp_b_held_out_checks.json").is_file()

        # Frozen: 2 (sup clean/noisy) + 2 (joint clean/noisy) = 4 rows
        # Clean stream: 4 checkpoints (0, 100, 200, 230) * 2 views = 8 rows
        # Noisy stream: 4 checkpoints (0, 100, 200, 230) * 2 views = 8 rows
        assert len(rows) == 20

        with open(out_csv, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            assert tuple(reader.fieldnames or ()) == HELD_OUT_CSV_COLUMNS

        # Both online_ttt streams start at step 0 matching joint_frozen exactly (fresh checkpoint)
        jf_clean = next(r for r in rows if r["condition"] == "joint_frozen" and r["held_out_condition"] == "clean")
        onl_clean_0 = next(
            r for r in rows
            if r["condition"] == "online_ttt"
            and r["adaptation_stream_condition"] == "clean"
            and r["adaptation_images_processed"] == 0
            and r["held_out_condition"] == "clean"
        )
        onl_noisy_0 = next(
            r for r in rows
            if r["condition"] == "online_ttt"
            and r["adaptation_stream_condition"] == "noisy"
            and r["adaptation_images_processed"] == 0
            and r["held_out_condition"] == "clean"
        )
        assert onl_clean_0["held_out_accuracy"] == jf_clean["held_out_accuracy"]
        assert onl_noisy_0["held_out_accuracy"] == jf_clean["held_out_accuracy"]

    def test_cli_check_splits_only_via_ttt_cifar100_entrypoint(self, tmp_path: Path):
        split_path = tmp_path / "split.json"
        split_path.write_text(
            json.dumps({
                "adaptation_ids": ["test:49", "test:141"],
                "held_out_ids": ["test:161", "test:816"],
            })
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.ttt_cifar100",
                "--held-out",
                "--split-file",
                str(split_path),
                "--check-splits-only",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        parsed = json.loads(proc.stdout)
        assert parsed["adaptation_count"] == 2
        assert parsed["held_out_count"] == 2
        assert parsed["held_out_examined_in_prior_exp_b"] is True

    def test_real_checkpoints_and_cached_cifar100_mini_pilot(self, tmp_path: Path):
        sup_ckpt = Path("checkpoints/exp_b_supervised_seed42.pt")
        joint_ckpt = Path("checkpoints/exp_b_joint_seed42.pt")
        test_mem_path = Path("data/exp_b_test_membership.json")
        if not (sup_ckpt.is_file() and joint_ckpt.is_file() and test_mem_path.is_file()):
            pytest.skip("Seed-42 checkpoints or exp_b_test_membership.json not present")

        test_rows = json.loads(test_mem_path.read_text(encoding="utf-8"))["test"]
        # Use 10 images for adaptation (eval_interval=5 -> [0, 5, 10]) and 10 disjoint held-out images
        split_path = tmp_path / "mini_real_split.json"
        split_path.write_text(
            json.dumps({
                "source_split": "test",
                "adaptation": test_rows[:10],
                "held_out": test_rows[10:20],
            })
        )
        out_csv = tmp_path / "mini_real_held_out.csv"
        rows, report = run_held_out_experiment(
            split_file=split_path,
            output_csv=out_csv,
            supervised_ckpt=sup_ckpt,
            joint_ckpt=joint_ckpt,
            eval_interval=5,
            eval_batch_size=5,
            device="cpu",
        )
        assert report["all_checks_passed"] is True
        assert report["split_checks"]["held_out_examined_in_prior_exp_b"] is True
        # Frozen (2 sup + 2 joint = 4) + clean stream (3 checkpoints * 2) + noisy stream (3 checkpoints * 2) = 16 rows
        assert len(rows) == 16
        assert all(0.0 <= r["held_out_accuracy"] <= 1.0 for r in rows)


# ==============================================================================
# 5. Regression Tests for Review Findings (Splits, Config, Streaming, Edge Cases)
# ==============================================================================


class _DistinctSyntheticCIFAR:
    """Synthetic dataset with constant per-split RGB color to detect split mix-ups."""

    def __init__(self, num_samples: int, fill_rgb: tuple[int, int, int]):
        self.num_samples = num_samples
        self.fill_rgb = fill_rgb

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "img": Image.new("RGB", (32, 32), color=self.fill_rgb),
            "fine_label": idx % 20,
        }


class TestRegressionAndEdgeCases:
    def test_split_loader_does_not_cross_contaminate_train_and_test_splits(
        self, tmp_path: Path, monkeypatch
    ):
        train_ds = _DistinctSyntheticCIFAR(50, fill_rgb=(10, 20, 30))
        test_ds = _DistinctSyntheticCIFAR(50, fill_rgb=(200, 210, 220))

        def fake_load_dataset(name, split, cache_dir=None):
            return train_ds if split == "train" else test_ds

        monkeypatch.setattr(data_mod, "load_dataset", fake_load_dataset)

        # Single-split dataset with missing requested split must NOT reuse the wrong split
        subset_test = CIFAR100SplitSubsetDataset(
            [("test", 0), ("test", 1)],
            datasets_by_split={"train": train_ds},
        )
        img_test, _, sid_test = subset_test[0]
        assert sid_test == "test:0"
        assert torch.allclose(img_test[0], torch.full((32, 32), 200.0 / 255.0), atol=1e-4)

        # Adaptation on 'train', held-out on 'test' (without passing 'test' upfront)
        split_path = tmp_path / "cross_split.json"
        split_path.write_text(
            json.dumps({
                "adaptation_ids": ["train:0", "train:1"],
                "held_out_ids": ["test:2", "test:3"],
            }),
            encoding="utf-8",
        )
        adapt_loader, ho_clean, _, _ = get_exp_b_split_loaders(
            split_path,
            train_val_membership=None,
            prior_test_membership=None,
            eval_batch_size=2,
        )
        adapt_img, _, adapt_sid = next(iter(adapt_loader))
        ho_img, _, ho_sids = next(iter(ho_clean))
        assert adapt_sid == ("train:0",)
        assert list(ho_sids) == ["test:2", "test:3"]
        assert torch.allclose(adapt_img[0, 0], torch.full((32, 32), 10.0 / 255.0), atol=1e-4)
        assert torch.allclose(ho_img[0, 0], torch.full((32, 32), 200.0 / 255.0), atol=1e-4)

        # Mixed train+test IDs within both adaptation and held-out splits
        mixed_path = tmp_path / "mixed_split.json"
        mixed_path.write_text(
            json.dumps({
                "adaptation_ids": ["train:5", "test:6"],
                "held_out_ids": ["train:7", "test:8"],
            }),
            encoding="utf-8",
        )
        m_adapt, m_ho_clean, _, _ = get_exp_b_split_loaders(
            mixed_path,
            datasets_by_split={"train": train_ds, "test": test_ds},
            train_val_membership=None,
            prior_test_membership=None,
            eval_batch_size=2,
        )
        adapt_batches = list(m_adapt)
        assert adapt_batches[0][2] == ("train:5",)
        assert adapt_batches[1][2] == ("test:6",)
        assert torch.allclose(adapt_batches[0][0][0, 0], torch.full((32, 32), 10.0 / 255.0), atol=1e-4)
        assert torch.allclose(adapt_batches[1][0][0, 0], torch.full((32, 32), 200.0 / 255.0), atol=1e-4)
        ho_batch = next(iter(m_ho_clean))
        assert list(ho_batch[2]) == ["train:7", "test:8"]

    def test_load_exp_b_split_pair_handles_prefixed_memberships_and_rejects_bad_source_split(
        self, tmp_path: Path
    ):
        bad_split = tmp_path / "bad_source.json"
        bad_split.write_text(
            json.dumps({"source_split": "validation", "adaptation": [0], "held_out": [1]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Unsupported source_split"):
            load_exp_b_split_ids(bad_split, split_role="adaptation")

        bad_role_split = tmp_path / "bad_role_source.json"
        bad_role_split.write_text(
            json.dumps({"adaptation_source_split": "val", "adaptation": [0]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Unsupported source_split"):
            load_exp_b_split_ids(bad_role_split, split_role="adaptation")

        # String-prefixed train_val_membership and prior_test_membership
        tv_path = tmp_path / "tv_prefixed.json"
        tv_path.write_text(
            json.dumps({
                "source_split": "train",
                "train": ["train:10", "train:11"],
                "validation": ["train:12"],
            }),
            encoding="utf-8",
        )
        prior_path = tmp_path / "prior_prefixed.json"
        prior_path.write_text(
            json.dumps({"source_split": "test", "test": ["test:100", "test:101"]}),
            encoding="utf-8",
        )
        split_ok = tmp_path / "ok_split.json"
        split_ok.write_text(
            json.dumps({"adaptation_ids": ["test:200"], "held_out_ids": ["test:100", "test:201"]}),
            encoding="utf-8",
        )
        _, _, meta = load_exp_b_split_pair(
            split_ok, train_val_membership=tv_path, prior_test_membership=prior_path
        )
        assert meta["held_out_examined_in_prior_exp_b"] is True
        assert meta["held_out_prior_overlap_count"] == 1

        split_leak = tmp_path / "leak_prefixed.json"
        split_leak.write_text(
            json.dumps({"adaptation_ids": ["train:12"], "held_out_ids": ["test:202"]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="train/validation"):
            load_exp_b_split_pair(
                split_leak, train_val_membership=tv_path, prior_test_membership=prior_path
            )

    def test_capture_rng_state_checks_mps_when_available(self, monkeypatch):
        mps_tensor = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        monkeypatch.setattr(torch.mps, "get_rng_state", lambda: mps_tensor.clone())

        dev = torch.device("mps")
        before = _capture_rng_state(dev)
        assert "torch_mps" in before
        assert torch.equal(before["torch_mps"], mps_tensor)
        _assert_rng_state_unchanged(before, dev)

        mutated = torch.tensor([9, 2, 3, 4], dtype=torch.uint8)
        monkeypatch.setattr(torch.mps, "get_rng_state", lambda: mutated)
        with pytest.raises(RuntimeError, match="random-number state"):
            _assert_rng_state_unchanged(before, dev)

    def test_batched_two_tuple_adaptation_loader_and_lazy_streaming(self):
        torch.manual_seed(0)
        model = SmallResNetTTT(num_classes=100)
        two_tuple_ds = TensorDataset(torch.rand(7, 3, 32, 32), torch.arange(7) % 20)
        batched_loader = DataLoader(two_tuple_ds, batch_size=3, shuffle=False)

        ho_clean, ids_clean = prepare_held_out_batches(
            _make_loader([("test", 100 + i) for i in range(6)], batch_size=3), batch_size=3
        )
        ho_noisy, ids_noisy = prepare_held_out_batches(
            _make_loader(
                [("test", 100 + i) for i in range(6)], corruption="gaussian_noise", batch_size=3
            ),
            batch_size=3,
        )

        rows, audit = run_held_out_stream(
            model=model,
            adaptation_loader=batched_loader,
            held_out_clean_batches=ho_clean,
            held_out_clean_ids=ids_clean,
            held_out_noisy_batches=ho_noisy,
            held_out_noisy_ids=ids_noisy,
            condition="online_ttt",
            adaptation_stream_condition="clean",
            eval_interval=5,
            ttt_lr=0.001,
            device="cpu",
        )
        assert audit["checkpoints_evaluated"] == [0, 5, 7]
        assert audit["total_adaptation_images"] == 7
        assert len(rows) == 6

        # Verify lazy streaming does not pull past max_adapt_images
        pulled = 0

        def _gen():
            nonlocal pulled
            for idx in range(10):
                pulled += 1
                yield (torch.rand(1, 3, 32, 32), torch.tensor([idx % 20]), [f"test:{idx}"])

        _, audit_lazy = run_held_out_stream(
            model=SmallResNetTTT(num_classes=100),
            adaptation_loader=_gen(),
            held_out_clean_batches=ho_clean,
            held_out_clean_ids=ids_clean,
            held_out_noisy_batches=ho_noisy,
            held_out_noisy_ids=ids_noisy,
            condition="online_ttt",
            adaptation_stream_condition="clean",
            eval_interval=2,
            max_adapt_images=3,
            device="cpu",
        )
        assert pulled == 3
        assert audit_lazy["total_adaptation_images"] == 3
        assert audit_lazy["checkpoints_evaluated"] == [0, 2, 3]

    def test_ttt_lr_zero_records_zero_updates_and_preserves_weights(self):
        torch.manual_seed(11)
        model = SmallResNetTTT(num_classes=100)
        init_state = {k: v.clone() for k, v in model.state_dict().items()}
        adapt_loader = _make_loader([("test", i) for i in range(4)], batch_size=1)
        ho_clean, ids_clean = prepare_held_out_batches(
            _make_loader([("test", 200 + i) for i in range(4)], batch_size=4), batch_size=4
        )
        ho_noisy, ids_noisy = prepare_held_out_batches(
            _make_loader(
                [("test", 200 + i) for i in range(4)], corruption="gaussian_noise", batch_size=4
            ),
            batch_size=4,
        )

        rows, audit = run_held_out_stream(
            model=model,
            adaptation_loader=adapt_loader,
            held_out_clean_batches=ho_clean,
            held_out_clean_ids=ids_clean,
            held_out_noisy_batches=ho_noisy,
            held_out_noisy_ids=ids_noisy,
            condition="online_ttt",
            adaptation_stream_condition="clean",
            eval_interval=2,
            ttt_lr=0.0,
            ttt_steps=3,
            device="cpu",
        )
        assert audit["total_updates"] == 0
        assert audit["features_adapted"] is False
        assert all(r["adaptation_update_count"] == 0 for r in rows)
        for k, v in init_state.items():
            assert torch.equal(model.state_dict()[k], v)

    def test_overwrite_protection_checkpoint_metadata_and_missing_config(self, tmp_path: Path):
        sup_model = SmallResNetTTT(num_classes=100)
        joint_model = SmallResNetTTT(num_classes=100)
        sup_ckpt = tmp_path / "sup_42.pt"
        joint_ckpt = tmp_path / "joint_42.pt"
        torch.save({"model_state_dict": sup_model.state_dict(), "mode": "supervised", "seed": 42}, sup_ckpt)
        torch.save({"model_state_dict": joint_model.state_dict(), "mode": "joint", "seed": 42}, joint_ckpt)

        split_path = tmp_path / "split.json"
        split_path.write_text(
            json.dumps({
                "adaptation_ids": [f"test:{i}" for i in range(4)],
                "held_out_ids": [f"test:{10 + i}" for i in range(4)],
            })
        )
        synth = {"test": SyntheticCIFAR(num_samples=50)}

        # Nonexistent config must raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            run_held_out_experiment(
                split_file=split_path,
                config_path=tmp_path / "missing.yaml",
                output_csv=tmp_path / "out.csv",
                supervised_ckpt=sup_ckpt,
                joint_ckpt=joint_ckpt,
                datasets_by_split=synth,
                train_val_membership=None,
                prior_test_membership=None,
                device="cpu",
            )

        # Supervised checkpoint passed as joint_ckpt must be rejected
        with pytest.raises(ValueError, match="mode='supervised', expected 'joint'"):
            run_held_out_experiment(
                split_file=split_path,
                output_csv=tmp_path / "bad_joint.csv",
                supervised_ckpt=sup_ckpt,
                joint_ckpt=sup_ckpt,
                datasets_by_split=synth,
                train_val_membership=None,
                prior_test_membership=None,
                device="cpu",
            )

        # Mismatched seed metadata must be rejected
        with pytest.raises(ValueError, match="seed=42, expected seed=43"):
            run_held_out_experiment(
                split_file=split_path,
                output_csv=tmp_path / "bad_seed.csv",
                supervised_ckpt=sup_ckpt,
                joint_ckpt=joint_ckpt,
                seed=43,
                datasets_by_split=synth,
                train_val_membership=None,
                prior_test_membership=None,
                device="cpu",
            )

        # Overwrite protection on existing output CSV
        out_csv = tmp_path / "existing.csv"
        out_csv.write_text("sentinel\n", encoding="utf-8")
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            run_held_out_experiment(
                split_file=split_path,
                output_csv=out_csv,
                supervised_ckpt=sup_ckpt,
                joint_ckpt=joint_ckpt,
                datasets_by_split=synth,
                train_val_membership=None,
                prior_test_membership=None,
                device="cpu",
            )
        assert out_csv.read_text(encoding="utf-8") == "sentinel\n"

        # Explicit overwrite=True succeeds and writes interval/cumulative timing fields
        rows, report = run_held_out_experiment(
            split_file=split_path,
            output_csv=out_csv,
            supervised_ckpt=sup_ckpt,
            joint_ckpt=joint_ckpt,
            eval_interval=2,
            eval_batch_size=2,
            datasets_by_split=synth,
            train_val_membership=None,
            prior_test_membership=None,
            device="cpu",
            overwrite=True,
        )
        assert report["all_checks_passed"] is True
        assert "timing_semantics" in report
        assert all("adaptation_interval_time_seconds" in r for r in rows)
        assert all("cumulative_held_out_eval_time_seconds" in r for r in rows)

    def test_full_held_out_cli_honors_yaml_config_and_rejects_bad_optimizer(
        self, tmp_path: Path
    ):
        test_mem_path = Path("data/exp_b_test_membership.json")
        if not test_mem_path.is_file():
            pytest.skip("data/exp_b_test_membership.json not present")
        test_rows = json.loads(test_mem_path.read_text(encoding="utf-8"))["test"]

        torch.manual_seed(17)
        sup_ckpt = tmp_path / "custom_sup.pt"
        joint_ckpt = tmp_path / "custom_joint.pt"
        torch.save(
            {"model_state_dict": SmallResNetTTT(num_classes=100).state_dict(), "mode": "supervised", "seed": 17},
            sup_ckpt,
        )
        torch.save(
            {"model_state_dict": SmallResNetTTT(num_classes=100).state_dict(), "mode": "joint", "seed": 17},
            joint_ckpt,
        )

        tv_path = tmp_path / "tv.json"
        tv_path.write_text(json.dumps({"source_split": "train", "train": [0], "validation": [1]}))
        prior_path = tmp_path / "prior.json"
        prior_path.write_text(json.dumps({"source_split": "test", "test": [test_rows[6]]}))

        split_path = tmp_path / "split.json"
        split_path.write_text(
            json.dumps({
                "source_split": "test",
                "adaptation": test_rows[:6],
                "held_out": test_rows[6:12],
            })
        )
        out_csv = tmp_path / "cli_out.csv"

        cfg_data = {
            "adaptation": {
                "ttt_optimizer": "sgd",
                "ttt_momentum": 0.0,
                "ttt_steps": 1,
                "ttt_lr": 0.002,
                "test_batch_size": 1,
                "lambda_rot": 1.0,
            },
            "corruption": {
                "type": "gaussian_noise",
                "sigma": 0.08,
                "noise_seed": 77,
            },
            "held_out_evaluation": {
                "seed": 17,
                "eval_interval": 3,
                "eval_batch_size": 3,
                "conditions": ["joint_frozen", "online_ttt"],
                "adaptation_streams": ["noisy"],
                "held_out_views": ["clean"],
            },
            "paths": {
                "shared_c20_membership": str(tv_path),
                "prior_exp_b_test_membership": str(prior_path),
                "supervised_checkpoint": str(sup_ckpt),
                "joint_checkpoint": str(joint_ckpt),
                "output_csv": str(out_csv),
            },
        }
        cfg_path = tmp_path / "custom_held_out.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.ttt_cifar100",
                "--held-out",
                "--config",
                str(cfg_path),
                "--split-file",
                str(split_path),
                "--device",
                "cpu",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert out_csv.is_file()
        with open(out_csv, "r", newline="", encoding="utf-8") as handle:
            written = list(csv.DictReader(handle))
        # joint_frozen (1 checkpoint * 1 view) + online_ttt noisy (0, 3, 6 -> 3 checkpoints * 1 view) = 4 rows
        assert len(written) == 4
        assert {r["condition"] for r in written} == {"joint_frozen", "online_ttt"}
        assert {r["held_out_condition"] for r in written} == {"clean"}
        assert {r["adaptation_stream_condition"] for r in written if r["condition"] == "online_ttt"} == {"noisy"}
        assert all(int(r["seed"]) == 17 for r in written)

        checks = json.loads((tmp_path / "cli_out_checks.json").read_text(encoding="utf-8"))
        assert checks["settings"]["noise_seed"] == 77
        assert checks["split_checks"]["held_out_examined_in_prior_exp_b"] is True

        # Unsupported optimizer in YAML must fail loudly
        bad_cfg = copy.deepcopy(cfg_data)
        bad_cfg["adaptation"]["ttt_optimizer"] = "adam"
        bad_cfg["paths"]["output_csv"] = str(tmp_path / "cli_bad_opt.csv")
        bad_cfg_path = tmp_path / "bad_opt.yaml"
        bad_cfg_path.write_text(yaml.safe_dump(bad_cfg), encoding="utf-8")
        proc_bad = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.ttt_cifar100",
                "--held-out",
                "--config",
                str(bad_cfg_path),
                "--split-file",
                str(split_path),
                "--device",
                "cpu",
            ],
            capture_output=True,
            text=True,
        )
        assert proc_bad.returncode != 0
        assert "ttt_optimizer" in proc_bad.stderr

        # Nonexistent --config via CLI must fail loudly
        proc_missing = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.ttt_cifar100",
                "--held-out",
                "--config",
                str(tmp_path / "does_not_exist.yaml"),
                "--split-file",
                str(split_path),
            ],
            capture_output=True,
            text=True,
        )
        assert proc_missing.returncode != 0
        assert "Config file not found" in proc_missing.stderr

    def test_membership_key_validation_none_split_cache_and_null_yaml_values(
        self, tmp_path: Path, monkeypatch
    ):
        split_path = tmp_path / "split.json"
        split_path.write_text(
            json.dumps({"adaptation_ids": ["train:0", "train:1"], "held_out_ids": ["test:2", "test:3"]})
        )

        # Malformed train_val_membership / prior_test_membership missing required keys must fail
        with pytest.raises(ValueError, match="'train' and 'validation'"):
            load_exp_b_split_pair(
                split_path,
                train_val_membership={"source_split": "train", "train_rows": [0], "val_rows": [1]},
                prior_test_membership=None,
            )
        with pytest.raises(ValueError, match="'test' or 'held_out'"):
            load_exp_b_split_pair(
                split_path,
                train_val_membership=None,
                prior_test_membership={"source_split": "test", "rows": [2]},
            )

        # datasets_by_split with None entry falls back to load_dataset rather than TypeError
        test_ds = _DistinctSyntheticCIFAR(10, fill_rgb=(50, 60, 70))
        monkeypatch.setattr(data_mod, "load_dataset", lambda *args, **kwargs: test_ds)
        subset = CIFAR100SplitSubsetDataset([("test", 0)], datasets_by_split={"test": None})
        assert subset[0][2] == "test:0"

        # 3-tuple adaptation batch with sample_ids=None works for batch_size > 1
        ho_clean, ids_clean = prepare_held_out_batches(
            _make_loader([("test", 10 + i) for i in range(4)], batch_size=2), batch_size=2
        )
        rows, audit = run_held_out_stream(
            model=SmallResNetTTT(num_classes=100),
            adaptation_loader=[(torch.rand(2, 3, 32, 32), torch.tensor([0, 1]), None)],
            held_out_clean_batches=ho_clean,
            held_out_clean_ids=ids_clean,
            held_out_views=["clean"],
            condition="online_ttt",
            adaptation_stream_condition="clean",
            eval_interval=2,
            device="cpu",
        )
        assert audit["total_adaptation_images"] == 2
        assert len(rows) == 2

        # Non-mapping YAML config fails cleanly in --check-splits-only
        list_cfg = tmp_path / "list.yaml"
        list_cfg.write_text("- bad\n", encoding="utf-8")
        proc_list = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.ttt_cifar100",
                "--held-out",
                "--config",
                str(list_cfg),
                "--split-file",
                str(split_path),
                "--check-splits-only",
            ],
            capture_output=True,
            text=True,
        )
        assert proc_list.returncode != 0
        assert "YAML mapping" in proc_list.stderr

        # Explicit YAML nulls fall back to defaults instead of crashing on int(None)/Path(None)
        sup_ckpt = tmp_path / "sup.pt"
        torch.save(
            {"model_state_dict": SmallResNetTTT(num_classes=100).state_dict(), "mode": "supervised", "seed": 42},
            sup_ckpt,
        )
        null_cfg = tmp_path / "nulls.yaml"
        null_cfg.write_text(
            yaml.safe_dump({
                "adaptation": {"ttt_lr": None},
                "held_out_evaluation": {"seed": None, "conditions": ["supervised_frozen"], "held_out_views": ["clean"]},
                "paths": {"output_csv": None},
            }),
            encoding="utf-8",
        )
        out_csv = tmp_path / "null_ok.csv"
        rows_null, _ = run_held_out_experiment(
            split_file=split_path,
            config_path=null_cfg,
            output_csv=out_csv,
            supervised_ckpt=sup_ckpt,
            datasets_by_split={"train": test_ds, "test": test_ds},
            train_val_membership=None,
            prior_test_membership=None,
            device="cpu",
        )
        assert len(rows_null) == 1
        assert rows_null[0]["seed"] == 42

    def test_yaml_null_or_empty_membership_still_enforces_leakage_and_empty_paths(
        self, tmp_path: Path, monkeypatch
    ):
        from src import ttt_held_out as ho_mod

        tv_path = tmp_path / "tv.json"
        tv_path.write_text(
            json.dumps({"source_split": "train", "train": [0, 1], "validation": [2]}),
            encoding="utf-8",
        )
        prior_path = tmp_path / "prior.json"
        prior_path.write_text(
            json.dumps({"source_split": "test", "test": [10]}),
            encoding="utf-8",
        )
        monkeypatch.setattr(ho_mod, "SHARED_C20_MEMBERSHIP_PATH", tv_path)
        monkeypatch.setattr(ho_mod, "EXP_B_TEST_MEMBERSHIP_PATH", prior_path)

        # Split leaking train:2 from default train/val membership and overlapping test:10 from prior
        leak_split = tmp_path / "leak.json"
        leak_split.write_text(
            json.dumps({"adaptation_ids": ["train:2"], "held_out_ids": ["test:10"]}),
            encoding="utf-8",
        )
        ok_split = tmp_path / "ok.json"
        ok_split.write_text(
            json.dumps({"adaptation_ids": ["test:20"], "held_out_ids": ["test:10"]}),
            encoding="utf-8",
        )

        sup_ckpt = tmp_path / "sup.pt"
        torch.save(
            {
                "model_state_dict": SmallResNetTTT(num_classes=100).state_dict(),
                "mode": "supervised",
                "seed": 42,
            },
            sup_ckpt,
        )
        synth = {"test": _DistinctSyntheticCIFAR(30, fill_rgb=(40, 50, 60))}

        for bad_val in (None, ""):
            cfg_path = tmp_path / f"cfg_{'null' if bad_val is None else 'empty'}.yaml"
            cfg_path.write_text(
                yaml.safe_dump({
                    "held_out_evaluation": {
                        "conditions": ["supervised_frozen"],
                        "held_out_views": ["clean"],
                    },
                    "paths": {
                        "shared_c20_membership": bad_val,
                        "prior_exp_b_test_membership": bad_val,
                        "output_csv": "",
                    },
                }),
                encoding="utf-8",
            )
            # run_held_out_experiment must still reject train/val leakage when YAML has null/""
            with pytest.raises(ValueError, match="train/validation"):
                run_held_out_experiment(
                    split_file=leak_split,
                    config_path=cfg_path,
                    output_csv=tmp_path / f"leak_{'n' if bad_val is None else 'e'}.csv",
                    supervised_ckpt=sup_ckpt,
                    datasets_by_split=synth,
                    device="cpu",
                )
            # Prior overlap must still be detected when YAML has null/""
            _, rep = run_held_out_experiment(
                split_file=ok_split,
                config_path=cfg_path,
                output_csv=tmp_path / f"ok_{'n' if bad_val is None else 'e'}.csv",
                supervised_ckpt=sup_ckpt,
                datasets_by_split=synth,
                device="cpu",
            )
            assert rep["split_checks"]["held_out_examined_in_prior_exp_b"] is True

            # --check-splits-only must also enforce leakage & prior checks when YAML has null/""
            with pytest.raises(ValueError, match="train/validation"):
                ho_mod.main([
                    "--config",
                    str(cfg_path),
                    "--split-file",
                    str(leak_split),
                    "--check-splits-only",
                ])

        # Empty output_csv in YAML falls back cleanly (not Path('.'))
        fallback_out = tmp_path / "fallback.csv"
        monkeypatch.chdir(tmp_path)
        (tmp_path / "results").mkdir()
        fallback_cfg = tmp_path / "fallback_out.yaml"
        fallback_cfg.write_text(
            yaml.safe_dump({
                "held_out_evaluation": {
                    "conditions": ["supervised_frozen"],
                    "held_out_views": ["clean"],
                },
                "paths": {"output_csv": ""},
            }),
            encoding="utf-8",
        )
        _, rep_fb = run_held_out_experiment(
            split_file=ok_split,
            config_path=fallback_cfg,
            supervised_ckpt=sup_ckpt,
            datasets_by_split=synth,
            train_val_membership=None,
            prior_test_membership=None,
            device="cpu",
        )
        assert Path(rep_fb["output_csv"]).name == "exp_b_held_out_pilot.csv"
        assert not fallback_out.exists()

        # Explicit empty string paths fail clearly
        for bad_kw in (
            {"split_file": ""},
            {"split_file": ok_split, "held_out_file": ""},
            {"split_file": ok_split, "output_csv": ""},
        ):
            with pytest.raises(ValueError, match="non-empty path"):
                run_held_out_experiment(
                    config_path=None,
                    supervised_ckpt=sup_ckpt,
                    datasets_by_split=synth,
                    train_val_membership=None,
                    prior_test_membership=None,
                    device="cpu",
                    **bad_kw,
                )

    def test_held_out_validates_and_honors_model_architecture_yaml(self, tmp_path: Path):
        split_path = tmp_path / "split.json"
        split_path.write_text(
            json.dumps({"adaptation_ids": ["test:0", "test:1"], "held_out_ids": ["test:2", "test:3"]}),
            encoding="utf-8",
        )
        synth = {"test": _DistinctSyntheticCIFAR(10, fill_rgb=(30, 60, 90))}

        # Custom non-default architecture (num_outputs=50, num_groups=8, dropout=0.4)
        custom_model = SmallResNetTTT(num_classes=50, num_groups=8, dropout=0.4)
        custom_ckpt = tmp_path / "custom_sup.pt"
        torch.save(
            {
                "model_state_dict": custom_model.state_dict(),
                "mode": "supervised",
                "seed": 42,
                "config": {"num_outputs": 50, "num_groups": 8, "dropout": 0.4},
            },
            custom_ckpt,
        )

        # Matching YAML works end-to-end
        good_cfg = tmp_path / "good_arch.yaml"
        good_cfg.write_text(
            yaml.safe_dump({
                "num_outputs": 50,
                "num_groups": 8,
                "dropout": 0.4,
                "held_out_evaluation": {
                    "conditions": ["supervised_frozen"],
                    "held_out_views": ["clean"],
                },
            }),
            encoding="utf-8",
        )
        rows, rep = run_held_out_experiment(
            split_file=split_path,
            config_path=good_cfg,
            output_csv=tmp_path / "good_arch.csv",
            supervised_ckpt=custom_ckpt,
            datasets_by_split=synth,
            train_val_membership=None,
            prior_test_membership=None,
            device="cpu",
        )
        assert len(rows) == 1 and rep["all_checks_passed"] is True

        # Mismatched num_groups in YAML vs checkpoint config fails fast
        bad_groups_cfg = tmp_path / "bad_groups.yaml"
        bad_groups_cfg.write_text(
            yaml.safe_dump({
                "num_outputs": 50,
                "num_groups": 16,
                "dropout": 0.4,
                "held_out_evaluation": {
                    "conditions": ["supervised_frozen"],
                    "held_out_views": ["clean"],
                },
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="num_groups"):
            run_held_out_experiment(
                split_file=split_path,
                config_path=bad_groups_cfg,
                output_csv=tmp_path / "bad_groups.csv",
                supervised_ckpt=custom_ckpt,
                datasets_by_split=synth,
                train_val_membership=None,
                prior_test_membership=None,
                device="cpu",
            )

        # Mismatched num_outputs in YAML vs checkpoint fails fast
        bad_outputs_cfg = tmp_path / "bad_outputs.yaml"
        bad_outputs_cfg.write_text(
            yaml.safe_dump({
                "num_outputs": 100,
                "num_groups": 8,
                "held_out_evaluation": {
                    "conditions": ["supervised_frozen"],
                    "held_out_views": ["clean"],
                },
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="num_outputs"):
            run_held_out_experiment(
                split_file=split_path,
                config_path=bad_outputs_cfg,
                output_csv=tmp_path / "bad_outputs.csv",
                supervised_ckpt=custom_ckpt,
                datasets_by_split=synth,
                train_val_membership=None,
                prior_test_membership=None,
                device="cpu",
            )

    def test_held_out_paths_config_split_and_checks_plus_edge_validation(
        self, tmp_path: Path, monkeypatch
    ):
        from src import ttt_held_out as ho_mod

        adapt_file = tmp_path / "adapt.json"
        ho_file = tmp_path / "ho.json"
        adapt_file.write_text(
            json.dumps({"adaptation_ids": ["test:0", "test:1"]}), encoding="utf-8"
        )
        ho_file.write_text(
            json.dumps({"held_out_ids": ["test:2", "test:3"]}), encoding="utf-8"
        )
        synth = {"test": _DistinctSyntheticCIFAR(10, fill_rgb=(20, 40, 60))}

        sup50 = tmp_path / "sup50.pt"
        jnt100 = tmp_path / "jnt100.pt"
        torch.save(
            {
                "model_state_dict": SmallResNetTTT(num_classes=50).state_dict(),
                "mode": "supervised",
                "seed": 42,
                "config": {"num_outputs": 50},
            },
            sup50,
        )
        torch.save(
            {
                "model_state_dict": SmallResNetTTT(num_classes=100).state_dict(),
                "mode": "joint",
                "seed": 42,
                "config": {"num_outputs": 100},
            },
            jnt100,
        )

        # 1. paths.split_file, paths.held_out_file, paths.output_checks honored via CLI without --split-file
        custom_out = tmp_path / "out.csv"
        custom_chk = tmp_path / "nested" / "custom_checks.json"
        cfg_paths = tmp_path / "paths_cfg.yaml"
        cfg_paths.write_text(
            yaml.safe_dump({
                "num_outputs": 50,
                "held_out_evaluation": {
                    "conditions": ["supervised_frozen"],
                    "held_out_views": ["clean"],
                },
                "paths": {
                    "split_file": str(adapt_file),
                    "held_out_file": str(ho_file),
                    "output_csv": str(custom_out),
                    "output_checks": str(custom_chk),
                    "supervised_checkpoint": str(sup50),
                    "shared_c20_membership": None,
                    "prior_exp_b_test_membership": None,
                },
            }),
            encoding="utf-8",
        )
        tv_p = tmp_path / "missing_tv.json"
        tv_p.write_text(
            json.dumps({"source_split": "train", "train": [0], "validation": [1]}),
            encoding="utf-8",
        )
        pr_p = tmp_path / "missing_pr.json"
        pr_p.write_text(
            json.dumps({"source_split": "test", "test": [9]}), encoding="utf-8"
        )
        monkeypatch.setattr(ho_mod, "SHARED_C20_MEMBERSHIP_PATH", tv_p)
        monkeypatch.setattr(ho_mod, "EXP_B_TEST_MEMBERSHIP_PATH", pr_p)

        ho_mod.main(["--config", str(cfg_paths), "--check-splits-only"])
        _, rep_paths = run_held_out_experiment(
            config_path=cfg_paths,
            datasets_by_split=synth,
            device="cpu",
        )
        assert custom_out.is_file() and custom_chk.is_file()
        assert rep_paths["output_checks"] == str(custom_chk)
        assert str(adapt_file) in rep_paths["split_locations"]
        assert str(ho_file) in rep_paths["split_locations"]

        # 2. Empty strings in paths_cfg for split_file/held_out_file/output_checks fall back cleanly
        comb_split = tmp_path / "combined.json"
        comb_split.write_text(
            json.dumps({"adaptation_ids": ["test:4"], "held_out_ids": ["test:5"]}),
            encoding="utf-8",
        )
        empty_paths_cfg = tmp_path / "empty_paths.yaml"
        empty_paths_cfg.write_text(
            yaml.safe_dump({
                "num_outputs": 50,
                "held_out_evaluation": {
                    "conditions": ["supervised_frozen"],
                    "held_out_views": ["clean"],
                },
                "paths": {
                    "split_file": "",
                    "held_out_file": "",
                    "output_checks": "",
                    "output_csv": str(tmp_path / "fb.csv"),
                },
            }),
            encoding="utf-8",
        )
        _, rep_empty = run_held_out_experiment(
            split_file=comb_split,
            config_path=empty_paths_cfg,
            supervised_ckpt=sup50,
            datasets_by_split=synth,
            train_val_membership=None,
            prior_test_membership=None,
            device="cpu",
        )
        assert Path(rep_empty["output_checks"]).name == "fb_checks.json"

        # 3. Path("") rejected cleanly, float test_batch_size=1.5 rejected, config_path=None works for 50-out
        with pytest.raises(ValueError, match="non-empty path"):
            run_held_out_experiment(
                split_file=Path(""),
                config_path=None,
                output_csv=tmp_path / "e1.csv",
                supervised_ckpt=sup50,
                datasets_by_split=synth,
                train_val_membership=None,
                prior_test_membership=None,
            )

        bad_tbs_cfg = tmp_path / "tbs.yaml"
        bad_tbs_cfg.write_text(
            yaml.safe_dump({
                "adaptation": {"test_batch_size": 1.5},
                "held_out_evaluation": {
                    "conditions": ["supervised_frozen"],
                    "held_out_views": ["clean"],
                },
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="test_batch_size"):
            run_held_out_experiment(
                split_file=comb_split,
                config_path=bad_tbs_cfg,
                output_csv=tmp_path / "e2.csv",
                supervised_ckpt=sup50,
                datasets_by_split=synth,
                train_val_membership=None,
                prior_test_membership=None,
            )

        rows_no_cfg, _ = run_held_out_experiment(
            split_file=comb_split,
            config_path=None,
            output_csv=tmp_path / "no_cfg_50.csv",
            supervised_ckpt=sup50,
            conditions=["supervised_frozen"],
            held_out_views=["clean"],
            datasets_by_split=synth,
            train_val_membership=None,
            prior_test_membership=None,
            device="cpu",
        )
        assert len(rows_no_cfg) == 1

        with pytest.raises(ValueError, match="num_outputs=100, expected 50"):
            run_held_out_experiment(
                split_file=comb_split,
                config_path=None,
                output_csv=tmp_path / "mixed_out.csv",
                supervised_ckpt=sup50,
                joint_ckpt=jnt100,
                conditions=["supervised_frozen", "joint_frozen"],
                held_out_views=["clean"],
                datasets_by_split=synth,
                train_val_membership=None,
                prior_test_membership=None,
                device="cpu",
            )

    def test_khoi_draft_split_and_held_out_plot(self, tmp_path: Path):
        from results.generate_plot import generate_held_out_plot

        draft_split = Path("data/ttt_exp_b_held_out_draft.json")
        if not draft_split.is_file():
            pytest.skip("data/ttt_exp_b_held_out_draft.json not present")

        adapt_refs, held_out_refs, meta = load_exp_b_split_pair(draft_split)
        assert len(adapt_refs) == 1500
        assert len(held_out_refs) == 500
        assert meta["adaptation_count"] == 1500
        assert meta["held_out_count"] == 500
        assert meta["held_out_examined_in_prior_exp_b"] is True
        assert meta["held_out_prior_overlap_count"] == 500

        csv_path = tmp_path / "held_out_plot_input.csv"
        rows = [
            {
                "experiment": "B",
                "condition": "supervised_frozen",
                "seed": "42",
                "source_checkpoint": "sup.pt",
                "code_version": "test",
                "adaptation_stream_condition": "none",
                "held_out_condition": "clean",
                "adaptation_images_processed": "0",
                "held_out_image_count": "500",
                "correct_prediction_count": "315",
                "held_out_accuracy": "0.63",
                "adaptation_update_count": "0",
                "adaptation_time_seconds": "0.0",
                "adaptation_interval_time_seconds": "0.0",
                "held_out_eval_time_seconds": "0.1",
                "cumulative_held_out_eval_time_seconds": "0.1",
                "config_path": "cfg.yaml",
                "split_locations": str(draft_split),
                "hardware": "cpu",
            },
            {
                "experiment": "B",
                "condition": "joint_frozen",
                "seed": "42",
                "source_checkpoint": "jnt.pt",
                "code_version": "test",
                "adaptation_stream_condition": "none",
                "held_out_condition": "noisy",
                "adaptation_images_processed": "0",
                "held_out_image_count": "500",
                "correct_prediction_count": "173",
                "held_out_accuracy": "0.346",
                "adaptation_update_count": "0",
                "adaptation_time_seconds": "0.0",
                "adaptation_interval_time_seconds": "0.0",
                "held_out_eval_time_seconds": "0.1",
                "cumulative_held_out_eval_time_seconds": "0.1",
                "config_path": "cfg.yaml",
                "split_locations": str(draft_split),
                "hardware": "cpu",
            },
            {
                "experiment": "B",
                "condition": "online_ttt",
                "seed": "42",
                "source_checkpoint": "jnt.pt",
                "code_version": "test",
                "adaptation_stream_condition": "noisy",
                "held_out_condition": "noisy",
                "adaptation_images_processed": "100",
                "held_out_image_count": "500",
                "correct_prediction_count": "190",
                "held_out_accuracy": "0.38",
                "adaptation_update_count": "100",
                "adaptation_time_seconds": "0.8",
                "adaptation_interval_time_seconds": "0.8",
                "held_out_eval_time_seconds": "0.1",
                "cumulative_held_out_eval_time_seconds": "0.2",
                "config_path": "cfg.yaml",
                "split_locations": str(draft_split),
                "hardware": "cpu",
            },
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HELD_OUT_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        out_png = tmp_path / "held_out_plot.png"
        generate_held_out_plot(str(csv_path), str(out_png))
        assert out_png.is_file() and out_png.stat().st_size > 0
