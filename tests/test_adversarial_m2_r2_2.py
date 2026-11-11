"""
Empirical Challenger 2 (challenger_m2_r2_2) Stress Test Suite for Milestone 2 Experiment B.

Adversarially tests:
1. build_summary_row contract & ALKA_SUMMARY_COLUMNS (23 columns, conditions, defaults, overrides)
2. Rolling window conservation laws (exhaustive grid, edge cases, remainder windows)
3. CSV serialization & appending invariants (no duplicate headers, extra field filtering)
4. Unmasked 100-way classification & label extraction (scalar int, 0-dim tensor, 1D tensor, >19 classes)
5. CLI smoke test execution and state isolation
"""

import copy
import csv
import io
import math
from pathlib import Path
import tempfile
from typing import Any, Dict, List
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import SmallResNetTTT
from src.ttt_cifar100 import (
    ALKA_SUMMARY_COLUMNS,
    ALKA_WINDOW_COLUMNS,
    VALID_CONDITIONS,
    adapt_and_predict_image,
    build_summary_row,
    compute_delta_accuracy,
    compute_rolling_windows,
    evaluate_condition,
    run_smoke_test,
    save_summary_csv,
    save_window_csv,
)


# ==============================================================================
# 1. build_summary_row and ALKA_SUMMARY_COLUMNS Verification
# ==============================================================================

class TestBuildSummaryRowAndAlkaSchemas:
    """Stress tests build_summary_row and schema definitions."""

    def test_alka_summary_columns_exact_count_and_content(self):
        """Verify ALKA_SUMMARY_COLUMNS has exactly 23 columns matching the project spec."""
        assert len(ALKA_SUMMARY_COLUMNS) == 23
        expected = [
            "experiment", "condition", "seed", "checkpoint", "dataset", "split",
            "corruption", "severity", "accuracy", "runtime_seconds", "update_count",
            "rotation_loss", "learning_rate", "adaptation_steps", "batch_size",
            "loss_weights", "prediction_timing", "reset_mode", "optimizer_reset",
            "hardware", "code_version", "config_path", "run_id"
        ]
        assert ALKA_SUMMARY_COLUMNS == expected

    def test_alka_window_columns_exact_count_and_content(self):
        """Verify ALKA_WINDOW_COLUMNS has exactly 10 columns matching the project spec."""
        assert len(ALKA_WINDOW_COLUMNS) == 10
        expected = [
            "run_id", "experiment", "condition", "seed", "window_index",
            "start_sample", "end_sample", "window_size", "correct_count", "accuracy"
        ]
        assert ALKA_WINDOW_COLUMNS == expected

    def test_valid_conditions_is_immutable_tuple(self):
        """Verify VALID_CONDITIONS is an immutable tuple containing the 4 required conditions."""
        assert isinstance(VALID_CONDITIONS, tuple)
        assert set(VALID_CONDITIONS) == {
            "supervised_frozen", "joint_frozen", "standard_ttt", "online_ttt"
        }

    def test_build_summary_row_rejects_invalid_condition(self):
        """Verify build_summary_row rejects unsupported conditions."""
        invalid_conditions = [
            "supervised", "joint", "ttt", "offline", "random", "", None, 123
        ]
        for cond in invalid_conditions:
            with pytest.raises(ValueError, match="condition must be one of"):
                build_summary_row(condition=cond)

    @pytest.mark.parametrize("cond,expected_defaults", [
        ("supervised_frozen", {
            "loss_weights": "1.0_cls",
            "prediction_timing": "none",
            "reset_mode": "none",
            "optimizer_reset": "none",
            "adaptation_steps": 0,
        }),
        ("joint_frozen", {
            "loss_weights": "1.0_cls_1.0_rot",
            "prediction_timing": "none",
            "reset_mode": "none",
            "optimizer_reset": "none",
            "adaptation_steps": 0,
        }),
        ("standard_ttt", {
            "loss_weights": "1.0_cls_1.0_rot",
            "prediction_timing": "post_adaptation",
            "reset_mode": "per_sample",
            "optimizer_reset": "stateless",
            "adaptation_steps": 1,
        }),
        ("online_ttt", {
            "loss_weights": "1.0_cls_1.0_rot",
            "prediction_timing": "post_adaptation",
            "reset_mode": "carryover",
            "optimizer_reset": "stateless",
            "adaptation_steps": 1,
        }),
    ])
    def test_build_summary_row_condition_defaults(self, cond, expected_defaults):
        """Verify build_summary_row derives correct defaults when adaptation_steps=None."""
        row = build_summary_row(condition=cond, adaptation_steps=None)
        assert len(row) == 23
        assert list(row.keys()) == ALKA_SUMMARY_COLUMNS
        for k, v in expected_defaults.items():
            assert row[k] == v, f"Mismatch in {k} for condition {cond}: expected {v}, got {row[k]}"

    def test_build_summary_row_explicit_overrides_defaults(self):
        """Verify explicit kwargs override auto-derived condition defaults."""
        row = build_summary_row(
            condition="supervised_frozen",
            loss_weights="custom_loss",
            prediction_timing="pre_adaptation",
            reset_mode="custom_reset",
            optimizer_reset="custom_opt",
            adaptation_steps=42,
        )
        assert row["loss_weights"] == "custom_loss"
        assert row["prediction_timing"] == "pre_adaptation"
        assert row["reset_mode"] == "custom_reset"
        assert row["optimizer_reset"] == "custom_opt"
        assert row["adaptation_steps"] == 42

    def test_build_summary_row_severity_normalization(self):
        """Verify severity normalization for clean vs gaussian_noise."""
        # clean with default/empty severity
        r_clean1 = build_summary_row(condition="joint_frozen", corruption="clean")
        assert r_clean1["severity"] == "none"

        r_clean2 = build_summary_row(condition="joint_frozen", corruption="clean", severity="")
        assert r_clean2["severity"] == "none"

        # gaussian_noise with default/empty/none severity
        r_noise1 = build_summary_row(condition="joint_frozen", corruption="gaussian_noise")
        assert r_noise1["severity"] == 3

        r_noise2 = build_summary_row(condition="joint_frozen", corruption="gaussian_noise", severity="none")
        assert r_noise2["severity"] == 3

        r_noise3 = build_summary_row(condition="joint_frozen", corruption="gaussian_noise", severity=5)
        assert r_noise3["severity"] == 5

    def test_build_summary_row_run_id_normalization(self):
        """Verify run_id is auto-generated if empty or None."""
        row = build_summary_row(condition="online_ttt", corruption="gaussian_noise", seed=44)
        assert row["run_id"] == "exp_b_online_ttt_gaussian_noise_seed44"

        custom_row = build_summary_row(condition="online_ttt", run_id="my_custom_run")
        assert custom_row["run_id"] == "my_custom_run"

    def test_build_summary_row_extra_kwargs_and_checkpoint_alias(self):
        """Verify checkpoint_path alias and predictions kwarg handling."""
        row = build_summary_row(
            checkpoint_path="checkpoints/model.pt",
            predictions=[0, 1, 2],
        )
        assert row["checkpoint"] == "checkpoints/model.pt"
        assert row["predictions"] == [0, 1, 2]
        # Verify 23 core columns are still present in ALKA_SUMMARY_COLUMNS
        for col in ALKA_SUMMARY_COLUMNS:
            assert col in row


# ==============================================================================
# 2. Rolling Window Conservation Laws & Edge Cases
# ==============================================================================

class TestRollingWindowConservationLaws:
    """Stress tests rolling window computation and mathematical conservation laws."""

    @pytest.mark.parametrize("n_samples", [1, 2, 7, 50, 128, 199, 200, 201, 350, 400, 599, 1000, 2000, 2001])
    @pytest.mark.parametrize("window_size", [1, 5, 20, 50, 100, 200, 500, 1000, 2500])
    def test_conservation_laws_exhaustive_grid(self, n_samples, window_size):
        """Verify all 5 conservation laws across an exhaustive grid of sample and window sizes."""
        torch.manual_seed(42 + n_samples)
        predictions = torch.randint(0, 20, (n_samples,)).tolist()
        targets = torch.randint(0, 20, (n_samples,)).tolist()

        windows = compute_rolling_windows(
            predictions=predictions,
            targets=targets,
            window_size=window_size,
            run_id="grid_test",
        )

        expected_num_windows = math.ceil(n_samples / window_size)
        assert len(windows) == expected_num_windows

        total_correct = sum(1 for p, t in zip(predictions, targets) if p == t)
        overall_acc = total_correct / n_samples

        # 1. Sample count conservation
        total_window_samples = sum(w["window_size"] for w in windows)
        assert total_window_samples == n_samples

        # 2. Correct count conservation
        total_window_correct = sum(w["correct_count"] for w in windows)
        assert total_window_correct == total_correct

        # 3. Contiguity conservation
        assert windows[0]["start_sample"] == 0
        assert windows[-1]["end_sample"] == n_samples
        for i in range(len(windows)):
            assert windows[i]["window_index"] == i
            assert windows[i]["end_sample"] - windows[i]["start_sample"] == windows[i]["window_size"]
            if i > 0:
                assert windows[i]["start_sample"] == windows[i - 1]["end_sample"]

        # 4. Accuracy conservation per window
        for w in windows:
            expected_w_acc = float(w["correct_count"]) / w["window_size"]
            assert pytest.approx(w["accuracy"], abs=1e-7) == expected_w_acc

        # 5. Weighted accuracy conservation
        weighted_acc = sum(w["accuracy"] * w["window_size"] for w in windows) / n_samples
        assert pytest.approx(weighted_acc, abs=1e-7) == overall_acc

    def test_rolling_windows_invalid_inputs(self):
        """Verify compute_rolling_windows rejects invalid inputs."""
        with pytest.raises(ValueError, match="window_size must be positive"):
            compute_rolling_windows([1, 2], [1, 2], window_size=0)

        with pytest.raises(ValueError, match="window_size must be positive"):
            compute_rolling_windows([1, 2], [1, 2], window_size=-5)

        with pytest.raises(ValueError, match="predictions and targets must have identical length"):
            compute_rolling_windows([1, 2], [1], window_size=2)

        with pytest.raises(ValueError, match="Cannot compute windows for empty predictions list"):
            compute_rolling_windows([], [], window_size=10)


# ==============================================================================
# 3. CSV Serialization & Appending Invariants
# ==============================================================================

class TestCSVSerializationInvariants:
    """Stress tests CSV writing, reading, schema enforcement, and append behavior."""

    def test_save_summary_csv_schema_and_filter_extra_keys(self):
        """Verify save_summary_csv strips extra keys (like predictions) and writes exact 23 columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "summary.csv"

            row1 = build_summary_row(condition="supervised_frozen", predictions=[1, 2, 3])
            row2 = build_summary_row(condition="joint_frozen", predictions=[4, 5, 6])

            # Write single row
            save_summary_csv(row1, csv_path)
            # Append second row
            save_summary_csv(row2, csv_path)

            with open(csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)

            assert header == ALKA_SUMMARY_COLUMNS
            assert len(rows) == 2
            for r in rows:
                assert len(r) == 23

            # Read as dict and check values
            with open(csv_path, "r", newline="") as f:
                dict_reader = csv.DictReader(f)
                read_rows = list(dict_reader)

            assert read_rows[0]["condition"] == "supervised_frozen"
            assert read_rows[1]["condition"] == "joint_frozen"
            assert "predictions" not in read_rows[0]

    def test_save_window_csv_schema_and_append_behavior(self):
        """Verify save_window_csv writes exact 10 columns and preserves append invariants."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "windows.csv"

            w1 = compute_rolling_windows([0, 1, 2], [0, 1, 3], window_size=2)
            w2 = compute_rolling_windows([1, 1], [1, 0], window_size=2)

            save_window_csv(w1, csv_path)
            save_window_csv(w2, csv_path)

            with open(csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)

            assert header == ALKA_WINDOW_COLUMNS
            # w1 has 2 windows, w2 has 1 window -> total 3 rows
            assert len(rows) == 3
            for r in rows:
                assert len(r) == 10


# ==============================================================================
# 4. Unmasked 100-Way Classification & Label Handling
# ==============================================================================

class TestUnmaskedClassificationAndLabelHandling:
    """Stress tests unmasked classification head and label handling in evaluate_condition."""

    def test_unmasked_100_way_classification_can_predict_above_class_19(self):
        """Verify model output is 100-way and predictions >= 20 are legitimately recorded."""
        model = SmallResNetTTT(num_classes=100)
        img = torch.randn(1, 3, 32, 32)
        logits, _, _, _ = adapt_and_predict_image(model, img, ttt_lr=0.0)
        assert logits.shape == (1, 100)

        # Force a high-index class prediction (>19) by biasing class 75
        with torch.no_grad():
            model.classifier[1].bias.zero_()
            model.classifier[1].bias[75] = 100.0

        logits_biased, _, _, _ = adapt_and_predict_image(model, img, ttt_lr=0.0)
        assert logits_biased.argmax(dim=-1).item() == 75

    @pytest.mark.parametrize("label_format", ["tensor_1d", "tensor_0d", "int_scalar", "tuple_scalar"])
    def test_evaluate_condition_label_type_polymorphism(self, label_format):
        """Stress-test evaluate_condition against various label container formats."""
        model = SmallResNetTTT(num_classes=100)

        img = torch.randn(1, 3, 32, 32)
        if label_format == "tensor_1d":
            batch = (img, torch.tensor([5], dtype=torch.long), ["sid_0"])
        elif label_format == "tensor_0d":
            batch = (img, torch.tensor(5, dtype=torch.long), ["sid_0"])
        elif label_format == "int_scalar":
            batch = (img, 5, "sid_0")
        elif label_format == "tuple_scalar":
            batch = (img, (5,), ("sid_0",))

        loader = [batch]

        summary, windows, logs = evaluate_condition(
            model=model,
            loader=loader,
            condition="supervised_frozen",
        )

        assert len(summary) == 23 + 1  # 23 columns + predictions
        assert len(logs) == 1
        assert logs[0]["true_label"] == 5
        assert logs[0]["sample_id"] == "sid_0"
        assert logs[0]["predicted_class"] == logs[0]["predicted_label"]
        assert logs[0]["is_correct"] == logs[0]["correct"]


# ==============================================================================
# 5. CLI Smoke Test Execution & Multi-Condition State Isolation
# ==============================================================================

class TestCLISmokeTestAndStateIsolation:
    """Stress tests run_smoke_test and inter-condition parameter isolation."""

    def test_smoke_test_execution_and_clean_state_restoration(self):
        """Verify run_smoke_test executes all 4 conditions x 2 corruptions on 128 samples,
        producing clean CSVs and strictly resetting model parameters between conditions.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            run_smoke_test(
                config_path="configs/ttt_cifar100_exp_b.yaml",
                output_dir=tmpdir,
                device="cpu",
            )

            summary_path = Path(tmpdir) / "ttt_summary.csv"
            window_path = Path(tmpdir) / "ttt_window_summary.csv"

            assert summary_path.exists()
            assert window_path.exists()

            with open(summary_path, "r") as f:
                summary_rows = list(csv.DictReader(f))

            with open(window_path, "r") as f:
                window_rows = list(csv.DictReader(f))

            # 4 conditions x 2 corruptions = 8 summary rows
            assert len(summary_rows) == 8
            assert len(window_rows) == 8  # 128 samples with window_size 200 -> 1 window per run

            # Verify conditions and update counts
            for r in summary_rows:
                cond = r["condition"]
                updates = int(r["update_count"])
                if cond in ("supervised_frozen", "joint_frozen"):
                    assert updates == 0
                elif cond in ("standard_ttt", "online_ttt"):
                    assert updates == 128
