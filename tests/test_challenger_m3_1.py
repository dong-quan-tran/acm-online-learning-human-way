"""Adversarial Verification & Empirical Stress Tests for Milestone 3 (Exp B).

Authored by Challenger 1.
Empirically audits:
1. Window Accuracy Conservation Law (sum of window correct counts == round(accuracy * 2000))
2. Window partitioning (contiguous, exact size 200, exactly partitions [0, 2000))
3. Single recorded prediction per test image (no dropped or duplicate samples)
4. Accuracy and metric bounds check (update counts, non-negative rotation loss, schemas)
5. Live empirical checkpoint reproduction (re-running evaluation on actual checkpoints)
"""

import csv
import json
import math
from pathlib import Path
import pytest
import torch

from src.data import (
    EXP_B_TEST_MEMBERSHIP_PATH,
    SHARED_C20_MEMBERSHIP_PATH,
    apply_gaussian_noise,
    get_exp_b_loader,
)
from src.models import SmallResNetTTT
from src.ttt_cifar100 import (
    ALKA_SUMMARY_COLUMNS,
    ALKA_WINDOW_COLUMNS,
    VALID_CONDITIONS,
    compute_rolling_windows,
    evaluate_condition,
    load_model,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_CSV = RESULTS_DIR / "ttt_summary.csv"
WINDOW_CSV = RESULTS_DIR / "ttt_window_summary.csv"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    assert path.is_file(), f"File does not exist: {path}"
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


class TestWindowAccuracyConservationLaw:
    """Audit conservation law: sum(window_correct) == round(acc * 2000)."""

    def test_summary_and_window_files_exist(self):
        assert SUMMARY_CSV.is_file(), f"Missing {SUMMARY_CSV}"
        assert WINDOW_CSV.is_file(), f"Missing {WINDOW_CSV}"

    def test_window_conservation_for_all_runs(self):
        summary_rows = load_csv_rows(SUMMARY_CSV)
        window_rows = load_csv_rows(WINDOW_CSV)

        assert len(summary_rows) == 8, f"Expected 8 summary rows, got {len(summary_rows)}"
        assert len(window_rows) == 80, f"Expected 80 window rows (8 runs * 10 windows), got {len(window_rows)}"

        # Group window rows by run_id
        windows_by_run: dict[str, list[dict[str, str]]] = {}
        for w in window_rows:
            run_id = w["run_id"]
            windows_by_run.setdefault(run_id, []).append(w)

        for s_row in summary_rows:
            run_id = s_row["run_id"]
            condition = s_row["condition"]
            corruption = s_row["corruption"]
            acc = float(s_row["accuracy"])

            assert run_id in windows_by_run, f"run_id '{run_id}' not found in window summary"
            run_windows = sorted(windows_by_run[run_id], key=lambda x: int(x["window_index"]))

            # Verify exactly 10 windows
            assert len(run_windows) == 10, f"{run_id}: expected 10 windows, got {len(run_windows)}"

            total_correct = 0
            total_samples = 0
            expected_total_correct = round(acc * 2000)

            for idx, w in enumerate(run_windows):
                w_idx = int(w["window_index"])
                assert w_idx == idx, f"{run_id}: window_index mismatch {w_idx} vs {idx}"
                assert w["condition"] == condition, f"{run_id}: condition mismatch {w['condition']} vs {condition}"
                assert int(w["seed"]) == int(s_row["seed"]), f"{run_id}: seed mismatch"

                start = int(w["start_sample"])
                end = int(w["end_sample"])
                size = int(w["window_size"])
                correct = int(w["correct_count"])
                w_acc = float(w["accuracy"])

                # Check window size and boundaries
                assert size == 200, f"{run_id} window {idx}: size is {size}, expected 200"
                assert start == idx * 200, f"{run_id} window {idx}: start {start} != {idx * 200}"
                assert end == (idx + 1) * 200, f"{run_id} window {idx}: end {end} != {(idx + 1) * 200}"
                assert end - start == size

                # Window accuracy formula check
                expected_w_acc = correct / size
                assert abs(w_acc - expected_w_acc) < 1e-6, (
                    f"{run_id} window {idx}: accuracy {w_acc} != {correct}/{size} = {expected_w_acc}"
                )

                total_correct += correct
                total_samples += size

            # Conservation checks
            assert total_samples == 2000, f"{run_id}: total samples {total_samples} != 2000"
            assert total_correct == expected_total_correct, (
                f"{run_id}: sum of window correct counts ({total_correct}) != "
                f"round(accuracy * 2000) ({expected_total_correct})"
            )
            # Check accuracy formula in summary
            summary_calculated_acc = total_correct / 2000.0
            assert abs(acc - summary_calculated_acc) < 1e-6, (
                f"{run_id}: summary accuracy {acc} != {total_correct} / 2000 = {summary_calculated_acc}"
            )


class TestSinglePredictionAndContiguity:
    """Audit single recorded prediction per test image and contiguous partitioning."""

    def test_window_contiguous_partition(self):
        window_rows = load_csv_rows(WINDOW_CSV)
        windows_by_run: dict[str, list[dict[str, str]]] = {}
        for w in window_rows:
            windows_by_run.setdefault(w["run_id"], []).append(w)

        for run_id, run_windows in windows_by_run.items():
            run_windows = sorted(run_windows, key=lambda x: int(x["window_index"]))
            assert int(run_windows[0]["start_sample"]) == 0, f"{run_id}: start_sample 0 is not 0"
            assert int(run_windows[-1]["end_sample"]) == 2000, f"{run_id}: end_sample 9 is not 2000"

            for i in range(len(run_windows) - 1):
                cur_end = int(run_windows[i]["end_sample"])
                nxt_start = int(run_windows[i + 1]["start_sample"])
                assert cur_end == nxt_start, (
                    f"{run_id}: Gap/overlap between window {i} (end={cur_end}) and {i+1} (start={nxt_start})"
                )

    def test_test_membership_properties(self):
        membership_path = EXP_B_TEST_MEMBERSHIP_PATH
        assert membership_path.is_file(), f"Missing test membership file: {membership_path}"

        with open(membership_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "test" in data
        test_indices = data["test"]
        assert len(test_indices) == 2000, f"Expected 2000 test indices, got {len(test_indices)}"
        assert len(set(test_indices)) == 2000, "Duplicate test indices found in membership!"
        assert test_indices == sorted(test_indices), "Test indices are not strictly in ascending order!"

        assert "classes" in data
        assert data["classes"] == list(range(20)), f"Classes must be 0..19, got {data['classes']}"
        assert data["num_samples"] == 2000


class TestMetricBoundsAndSchema:
    """Audit metric bounds, update counts, rotation loss, and schema integrity."""

    def test_summary_schema_columns(self):
        summary_rows = load_csv_rows(SUMMARY_CSV)
        for row in summary_rows:
            for col in ALKA_SUMMARY_COLUMNS:
                assert col in row, f"Missing column '{col}' in summary row"
                assert row[col] is not None and row[col] != "", f"Empty column '{col}' in summary row"

    def test_window_schema_columns(self):
        window_rows = load_csv_rows(WINDOW_CSV)
        for row in window_rows:
            for col in ALKA_WINDOW_COLUMNS:
                assert col in row, f"Missing column '{col}' in window row"
                assert row[col] is not None and row[col] != "", f"Empty column '{col}' in window row"

    def test_update_counts_by_condition(self):
        summary_rows = load_csv_rows(SUMMARY_CSV)
        for row in summary_rows:
            cond = row["condition"]
            updates = int(row["update_count"])
            adapt_steps = int(row["adaptation_steps"])

            if cond in ("supervised_frozen", "joint_frozen"):
                assert updates == 0, f"{cond}: update_count must be 0, got {updates}"
                assert adapt_steps == 0, f"{cond}: adaptation_steps must be 0, got {adapt_steps}"
                assert row["reset_mode"] == "none"
                assert row["optimizer_reset"] == "none"
                assert row["prediction_timing"] == "none"
            elif cond == "standard_ttt":
                assert updates == 2000, f"{cond}: update_count must be 2000, got {updates}"
                assert adapt_steps == 1, f"{cond}: adaptation_steps must be 1, got {adapt_steps}"
                assert row["reset_mode"] == "per_sample"
                assert row["optimizer_reset"] == "stateless"
                assert row["prediction_timing"] == "post_adaptation"
            elif cond == "online_ttt":
                assert updates == 2000, f"{cond}: update_count must be 2000, got {updates}"
                assert adapt_steps == 1, f"{cond}: adaptation_steps must be 1, got {adapt_steps}"
                assert row["reset_mode"] == "carryover"
                assert row["optimizer_reset"] == "stateless"
                assert row["prediction_timing"] == "post_adaptation"
            else:
                pytest.fail(f"Unexpected condition {cond}")

    def test_rotation_loss_positive_and_ordered(self):
        summary_rows = load_csv_rows(SUMMARY_CSV)
        by_corr: dict[str, dict[str, float]] = {}
        for row in summary_rows:
            rot_loss = float(row["rotation_loss"])
            assert rot_loss > 0.0, f"Rotation loss must be strictly positive, got {rot_loss}"
            by_corr.setdefault(row["corruption"], {})[row["condition"]] = rot_loss

        # In clean stream, joint training and TTT should produce lower rotation loss than supervised
        clean_losses = by_corr["clean"]
        assert clean_losses["supervised_frozen"] > clean_losses["joint_frozen"], (
            f"Supervised rot loss ({clean_losses['supervised_frozen']}) should be higher than "
            f"joint rot loss ({clean_losses['joint_frozen']})"
        )
        assert clean_losses["joint_frozen"] > clean_losses["standard_ttt"], (
            f"Joint frozen rot loss ({clean_losses['joint_frozen']}) should be higher than "
            f"standard TTT rot loss ({clean_losses['standard_ttt']})"
        )

        # In noise stream, supervised rotation loss should also be highest
        noise_losses = by_corr["gaussian_noise"]
        assert noise_losses["supervised_frozen"] > noise_losses["joint_frozen"]
        assert noise_losses["joint_frozen"] > noise_losses["online_ttt"]

    def test_hardware_and_metadata_invariants(self):
        summary_rows = load_csv_rows(SUMMARY_CSV)
        for row in summary_rows:
            assert row["dataset"] == "CIFAR-100-20"
            assert row["split"] == "test"
            assert row["experiment"] == "B"
            assert int(row["seed"]) == 42
            assert int(row["batch_size"]) == 1
            assert float(row["learning_rate"]) == 0.001
            assert Path(PROJECT_ROOT / row["checkpoint"]).is_file(), f"Checkpoint missing: {row['checkpoint']}"
            assert row["hardware"] in ("mps", "cuda", "cpu")
            assert len(row["code_version"]) >= 7


@pytest.fixture(scope="module")
def device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture(scope="module")
def clean_test_loader():
    return get_exp_b_loader(
        split="test",
        membership_path=SHARED_C20_MEMBERSHIP_PATH,
        test_membership_path=EXP_B_TEST_MEMBERSHIP_PATH,
        batch_size=64,
        shuffle=False,
        corruption="clean",
    )


@pytest.fixture(scope="module")
def noisy_test_loader():
    return get_exp_b_loader(
        split="test",
        membership_path=SHARED_C20_MEMBERSHIP_PATH,
        test_membership_path=EXP_B_TEST_MEMBERSHIP_PATH,
        batch_size=64,
        shuffle=False,
        corruption="gaussian_noise",
        noise_sigma=0.08,
        noise_seed=42,
    )


class TestEmpiricalLiveReproduction:
    """Empirically re-run inference on checkpoints to verify reported numbers."""


    def test_reproduce_supervised_frozen_clean(self, clean_test_loader, device):
        ckpt_path = CHECKPOINTS_DIR / "exp_b_supervised_seed42.pt"
        assert ckpt_path.is_file(), f"Missing checkpoint {ckpt_path}"
        model, _ = load_model(ckpt_path, device=device)

        summary_dict, _, _ = evaluate_condition(
            model=model,
            loader=clean_test_loader,
            condition="supervised_frozen",
            device=device,
            corruption="clean",
            seed=42,
        )

        reported_acc = 0.634  # from ttt_summary.csv
        actual_acc = summary_dict["accuracy"]
        assert actual_acc == pytest.approx(reported_acc, abs=1e-4), (
            f"Supervised frozen clean accuracy mismatch: got {actual_acc}, reported {reported_acc}"
        )

    def test_reproduce_joint_frozen_clean(self, clean_test_loader, device):
        ckpt_path = CHECKPOINTS_DIR / "exp_b_joint_seed42.pt"
        assert ckpt_path.is_file(), f"Missing checkpoint {ckpt_path}"
        model, _ = load_model(ckpt_path, device=device)

        summary_dict, _, _ = evaluate_condition(
            model=model,
            loader=clean_test_loader,
            condition="joint_frozen",
            device=device,
            corruption="clean",
            seed=42,
        )

        reported_acc = 0.6585  # from ttt_summary.csv
        actual_acc = summary_dict["accuracy"]
        assert actual_acc == pytest.approx(reported_acc, abs=1e-4), (
            f"Joint frozen clean accuracy mismatch: got {actual_acc}, reported {reported_acc}"
        )

    def test_reproduce_supervised_frozen_noise(self, noisy_test_loader, device):
        ckpt_path = CHECKPOINTS_DIR / "exp_b_supervised_seed42.pt"
        assert ckpt_path.is_file(), f"Missing checkpoint {ckpt_path}"
        model, _ = load_model(ckpt_path, device=device)

        summary_dict, _, _ = evaluate_condition(
            model=model,
            loader=noisy_test_loader,
            condition="supervised_frozen",
            device=device,
            corruption="gaussian_noise",
            severity=3,
            seed=42,
        )

        reported_acc = 0.3905  # from ttt_summary.csv
        actual_acc = summary_dict["accuracy"]
        assert actual_acc == pytest.approx(reported_acc, abs=1e-4), (
            f"Supervised frozen noise accuracy mismatch: got {actual_acc}, reported {reported_acc}"
        )

    def test_reproduce_joint_frozen_noise(self, noisy_test_loader, device):
        ckpt_path = CHECKPOINTS_DIR / "exp_b_joint_seed42.pt"
        assert ckpt_path.is_file(), f"Missing checkpoint {ckpt_path}"
        model, _ = load_model(ckpt_path, device=device)

        summary_dict, _, _ = evaluate_condition(
            model=model,
            loader=noisy_test_loader,
            condition="joint_frozen",
            device=device,
            corruption="gaussian_noise",
            severity=3,
            seed=42,
        )

        reported_acc = 0.3505  # from ttt_summary.csv
        actual_acc = summary_dict["accuracy"]
        assert actual_acc == pytest.approx(reported_acc, abs=1e-4), (
            f"Joint frozen noise accuracy mismatch: got {actual_acc}, reported {reported_acc}"
        )

    def test_reproduce_standard_ttt_window0_noise(self, device):
        ckpt_path = CHECKPOINTS_DIR / "exp_b_joint_seed42.pt"
        assert ckpt_path.is_file(), f"Missing checkpoint {ckpt_path}"
        model, _ = load_model(ckpt_path, device=device)

        base_loader = get_exp_b_loader(
            split="test",
            membership_path=SHARED_C20_MEMBERSHIP_PATH,
            test_membership_path=EXP_B_TEST_MEMBERSHIP_PATH,
            batch_size=1,
            shuffle=False,
            corruption="gaussian_noise",
            noise_sigma=0.08,
            noise_seed=42,
        )
        subset_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(base_loader.dataset, range(200)),
            batch_size=1,
            shuffle=False,
        )
        _, windows, _ = evaluate_condition(
            model=model,
            loader=subset_loader,
            condition="standard_ttt",
            ttt_lr=0.001,
            ttt_steps=1,
            window_size=200,
            device=device,
            corruption="gaussian_noise",
            seed=42,
        )

        # Window 0 reported in ttt_window_summary.csv: correct_count=80, accuracy=0.400
        assert windows[0]["correct_count"] == 80
        assert windows[0]["accuracy"] == pytest.approx(0.400, abs=1e-4)

    def test_reproduce_online_ttt_windows0_and_1_noise(self, device):
        ckpt_path = CHECKPOINTS_DIR / "exp_b_joint_seed42.pt"
        assert ckpt_path.is_file(), f"Missing checkpoint {ckpt_path}"
        model, _ = load_model(ckpt_path, device=device)

        base_loader = get_exp_b_loader(
            split="test",
            membership_path=SHARED_C20_MEMBERSHIP_PATH,
            test_membership_path=EXP_B_TEST_MEMBERSHIP_PATH,
            batch_size=1,
            shuffle=False,
            corruption="gaussian_noise",
            noise_sigma=0.08,
            noise_seed=42,
        )
        subset_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(base_loader.dataset, range(400)),
            batch_size=1,
            shuffle=False,
        )
        _, windows, _ = evaluate_condition(
            model=model,
            loader=subset_loader,
            condition="online_ttt",
            ttt_lr=0.001,
            ttt_steps=1,
            window_size=200,
            device=device,
            corruption="gaussian_noise",
            seed=42,
        )

        # Window 0 reported in ttt_window_summary.csv: correct_count=73, accuracy=0.365
        assert windows[0]["correct_count"] == 73
        assert windows[0]["accuracy"] == pytest.approx(0.365, abs=1e-4)

        # Window 1 reported in ttt_window_summary.csv: correct_count=81, accuracy=0.405
        assert windows[1]["correct_count"] == 81
        assert windows[1]["accuracy"] == pytest.approx(0.405, abs=1e-4)

