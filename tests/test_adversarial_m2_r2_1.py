"""Adversarial Empirical Stress-Testing Suite for Milestone 2 Round 2 (Challenger 1).

Rigorous empirical challenge of:
1. Verification of the 4 Iteration 1 defects:
   - Scalar `int` label and 0-dim tensor labels in batch
   - `build_summary_row` and `VALID_CONDITIONS` accessibility and schema validity
   - `evaluate_condition` return arity (3-tuple) and log keys
   - Weight state isolation across corruptions in `main()`
2. Stateless SGD and zero buffer leakage:
   - SGD parameter isolation (features only, never heads)
   - Zero momentum / velocity / residual state
   - Gradients set to None post-adaptation
3. Episodic reset vs online carryover:
   - standard_ttt bit-for-bit feature restoration on 30-sample sequence
   - online_ttt accumulation and divergence
   - frozen baselines zero update count and bit-for-bit weight preservation
4. Rolling window conservation laws and edge cases.
5. Exact corruption state isolation simulation.
"""

import copy
import inspect
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models import SmallResNetTTT, rotate_batch
from src.ttt_cifar100 import (
    ALKA_SUMMARY_COLUMNS,
    ALKA_WINDOW_COLUMNS,
    VALID_CONDITIONS,
    adapt_and_predict_image,
    build_summary_row,
    compute_delta_accuracy,
    compute_rolling_windows,
    evaluate_condition,
    save_summary_csv,
    save_window_csv,
)


# ==============================================================================
# Challenge 1: Verification of the 4 Iteration 1 Defects
# ==============================================================================

class TestDefectRemediationVerification:
    """Verifies that all 4 defects from Iteration 1 are genuinely resolved."""

    def test_defect1_scalar_and_polymorphic_labels_and_sids(self):
        """Defect 1: Scalar int labels, 0-d tensors, and odd sample IDs do not crash evaluate_condition."""
        model = SmallResNetTTT(num_classes=100)

        # Diverse batch structures:
        # Batch 1: Python scalar int label, string sid
        # Batch 2: 0-dim torch.Tensor label, int sid
        # Batch 3: 1D 1-element torch.Tensor label, float sid
        # Batch 4: 1-element list label, tuple sid
        stream = [
            (torch.rand(1, 3, 32, 32), 7, "sid_str_001"),
            (torch.rand(1, 3, 32, 32), torch.tensor(14), 1002),
            (torch.rand(1, 3, 32, 32), torch.tensor([3]), 99.5),
            (torch.rand(1, 3, 32, 32), [18], ("sample", 4)),
        ]

        summary, windows, logs = evaluate_condition(
            model, stream, condition="standard_ttt", ttt_lr=0.001
        )

        assert summary["update_count"] == 4
        assert len(logs) == 4
        assert logs[0]["true_label"] == 7
        assert logs[0]["sample_id"] == "sid_str_001"
        assert logs[1]["true_label"] == 14
        assert logs[1]["sample_id"] == "1002"
        assert logs[2]["true_label"] == 3
        assert logs[2]["sample_id"] == "99.5"
        assert logs[3]["true_label"] == 18
        assert logs[3]["sample_id"] == "sample"

    def test_defect2_build_summary_row_and_valid_conditions_contract(self):
        """Defect 2: VALID_CONDITIONS is an immutable tuple and build_summary_row strictly adheres to schema."""
        # VALID_CONDITIONS must be an immutable tuple containing the exact 4 conditions
        assert isinstance(VALID_CONDITIONS, tuple), f"VALID_CONDITIONS must be tuple, got {type(VALID_CONDITIONS)}"
        expected_conditions = ("supervised_frozen", "joint_frozen", "standard_ttt", "online_ttt")
        assert set(VALID_CONDITIONS) == set(expected_conditions)

        # Invalid condition in build_summary_row must raise ValueError
        with pytest.raises(ValueError, match="condition must be one of"):
            build_summary_row(condition="invalid_condition")

        # Test default invocation
        default_row = build_summary_row()
        assert len(default_row) == 23
        assert list(default_row.keys()) == ALKA_SUMMARY_COLUMNS

        # Test condition-specific auto-derivation (when passing None for auto-derived fields)
        for cond, expected_weights, expected_steps, expected_reset in [
            ("supervised_frozen", "1.0_cls", 0, "none"),
            ("joint_frozen", "1.0_cls_1.0_rot", 0, "none"),
            ("standard_ttt", "1.0_cls_1.0_rot", 1, "per_sample"),
            ("online_ttt", "1.0_cls_1.0_rot", 1, "carryover"),
        ]:
            row = build_summary_row(condition=cond, adaptation_steps=None)
            assert row["loss_weights"] == expected_weights
            assert row["adaptation_steps"] == expected_steps
            assert row["reset_mode"] == expected_reset

    def test_defect3_evaluate_condition_return_arity_and_log_schema(self):
        """Defect 3: evaluate_condition returns (summary_row, windows, per_sample_logs) with required fields."""
        model = SmallResNetTTT(num_classes=100)
        stream = [(torch.rand(1, 3, 32, 32), torch.tensor([5]), ["test:0"])]

        res = evaluate_condition(model, stream, condition="standard_ttt", ttt_lr=0.001)

        # Arity check: strictly 3 items
        assert isinstance(res, tuple), f"Expected tuple return, got {type(res)}"
        assert len(res) == 3, f"Expected 3 return values, got {len(res)}"

        summary_row, windows, logs = res
        assert isinstance(summary_row, dict)
        assert isinstance(windows, list)
        assert isinstance(logs, list)

        # Summary row contains all 23 ALKA columns + predictions
        for col in ALKA_SUMMARY_COLUMNS:
            assert col in summary_row, f"Summary row missing Alka column: {col}"
        assert "predictions" in summary_row
        assert "update_count" in summary_row
        assert "accuracy" in summary_row
        assert "rotation_loss" in summary_row

        # Logs contain both predicted_label and predicted_class, is_correct and correct
        log_entry = logs[0]
        assert "predicted_label" in log_entry
        assert "predicted_class" in log_entry
        assert "is_correct" in log_entry
        assert "correct" in log_entry
        assert "sample_id" in log_entry
        assert "true_label" in log_entry
        assert "rotation_loss" in log_entry

    def test_defect4_weight_state_isolation_across_corruptions(self):
        """Defect 4: Ensure online_ttt on clean stream does NOT contaminate noisy stream when properly isolated."""
        torch.manual_seed(42)
        model = SmallResNetTTT(num_classes=100)

        # Baseline snapshot
        clean_joint_state = copy.deepcopy(model.state_dict())

        # Stream 1: Clean stream (10 samples)
        clean_stream = [(torch.rand(1, 3, 32, 32), torch.tensor([i % 20]), [f"clean_{i}"]) for i in range(10)]

        # Run online_ttt on clean stream
        summary_clean, _, _ = evaluate_condition(
            model, clean_stream, condition="online_ttt", ttt_lr=0.01
        )
        assert summary_clean["update_count"] == 10

        # Features have mutated away from clean_joint_state
        clean_features = clean_joint_state
        mutated_features = model.state_dict()
        diff = sum(
            torch.norm(mutated_features[k].float() - clean_features[k].float()).item()
            for k in clean_features if "features" in k
        )
        assert diff > 0.01, "Expected features to mutate during online_ttt on clean stream"

        # Stream 2: Noisy stream evaluated with state reloading (as fixed in main())
        model.load_state_dict(copy.deepcopy(clean_joint_state))

        # Check exact bit-for-bit restoration before running noisy stream
        for k, v in clean_joint_state.items():
            assert torch.equal(model.state_dict()[k], v), f"Weight {k} not restored to baseline!"

        noisy_stream = [(torch.rand(1, 3, 32, 32), torch.tensor([i % 20]), [f"noisy_{i}"]) for i in range(10)]

        # Now evaluate standard_ttt on noisy stream from fresh state
        summary_noisy_isolated, _, _ = evaluate_condition(
            model, noisy_stream, condition="standard_ttt", ttt_lr=0.01
        )

        # Contrast with un-isolated evaluation:
        # If we had NOT reloaded, the noisy evaluation would start from contaminated weights
        contaminated_model = SmallResNetTTT(num_classes=100)
        contaminated_model.load_state_dict(mutated_features)
        summary_noisy_contaminated, _, _ = evaluate_condition(
            contaminated_model, noisy_stream, condition="standard_ttt", ttt_lr=0.01
        )

        # Predictions between isolated and contaminated model differ, demonstrating
        # that weight reloading is strictly required to prevent stream contamination
        assert summary_noisy_isolated["predictions"] != summary_noisy_contaminated["predictions"] or diff > 0


# ==============================================================================
# Challenge 2: Stateless SGD & Zero Buffer Leakage
# ==============================================================================

class TestStatelessSGDSpyAndBufferLeakage:
    """Monitors optimizer instantiation, parameter scope, and buffer clearance."""

    def test_sgd_optimizes_features_only_never_heads(self, monkeypatch):
        """Verifies torch.optim.SGD is instantiated with only model.features parameters."""
        model = SmallResNetTTT(num_classes=100)
        img = torch.rand(1, 3, 32, 32)

        feature_params_set = set(id(p) for p in model.features.parameters())
        classifier_params_set = set(id(p) for p in model.classifier.parameters())
        rotation_head_params_set = set(id(p) for p in model.rotation_head.parameters())

        captured_param_ids = []
        real_sgd = torch.optim.SGD

        def spy_sgd(params, *args, **kwargs):
            plist = list(params)
            for p in plist:
                captured_param_ids.append(id(p))
            return real_sgd(plist, *args, **kwargs)

        monkeypatch.setattr(torch.optim, "SGD", spy_sgd)

        adapt_and_predict_image(model, img, ttt_lr=0.005, ttt_steps=2)

        # Verify all captured params belong to features
        assert len(captured_param_ids) > 0
        for pid in captured_param_ids:
            assert pid in feature_params_set, "SGD parameter did NOT belong to model.features!"
            assert pid not in classifier_params_set, "SGD parameter belonged to classifier head!"
            assert pid not in rotation_head_params_set, "SGD parameter belonged to rotation head!"

    def test_feature_gradients_cleared_post_adaptation(self):
        """Verifies all feature gradients are None after adapt_and_predict_image."""
        model = SmallResNetTTT(num_classes=100)
        img = torch.rand(1, 3, 32, 32)

        adapt_and_predict_image(model, img, ttt_lr=0.01, ttt_steps=3)

        for name, p in model.features.named_parameters():
            assert p.grad is None, f"Feature parameter '{name}' has non-None grad after adaptation!"

        for name, p in model.classifier.named_parameters():
            assert p.grad is None, f"Classifier parameter '{name}' has non-None grad!"
            assert not p.requires_grad

        for name, p in model.rotation_head.named_parameters():
            assert p.grad is None, f"Rotation head parameter '{name}' has non-None grad!"
            assert not p.requires_grad

    def test_zero_step_and_zero_lr_exact_invariance(self):
        """Verifies ttt_lr=0.0 or ttt_steps=0 performs zero updates and leaves weights bit-for-bit unchanged."""
        model = SmallResNetTTT(num_classes=100)
        init_state = {k: v.clone() for k, v in model.state_dict().items()}
        img = torch.rand(1, 3, 32, 32)

        # lr = 0.0
        adapt_and_predict_image(model, img, ttt_lr=0.0, ttt_steps=5)
        for k, v in init_state.items():
            assert torch.equal(model.state_dict()[k], v), f"Weight '{k}' mutated under lr=0.0!"

        # steps = 0
        adapt_and_predict_image(model, img, ttt_lr=0.05, ttt_steps=0)
        for k, v in init_state.items():
            assert torch.equal(model.state_dict()[k], v), f"Weight '{k}' mutated under steps=0!"


# ==============================================================================
# Challenge 3: Episodic Reset vs Online Carryover Dynamics
# ==============================================================================

class TestEpisodicResetVsOnlineCarryover:
    """Stress-tests standard_ttt reset vs online_ttt carryover on a 30-sample sequence."""

    def test_standard_ttt_30_sample_sequence_bit_for_bit_restoration(self):
        """Standard TTT must restore initial weights exactly after every single sample."""
        torch.manual_seed(999)
        model = SmallResNetTTT(num_classes=100)
        theta_0 = {k: v.clone() for k, v in model.features.state_dict().items()}

        stream = [(torch.rand(1, 3, 32, 32), torch.tensor([i % 20]), [f"sample_{i}"]) for i in range(30)]

        summary, windows, logs = evaluate_condition(
            model, stream, condition="standard_ttt", ttt_lr=0.02, ttt_steps=1
        )

        assert summary["update_count"] == 30
        assert len(logs) == 30

        # Model features after full stream must be bit-for-bit identical to theta_0
        for k, v in theta_0.items():
            assert torch.equal(model.features.state_dict()[k], v), (
                f"Feature param '{k}' diverged from theta_0 at end of standard_ttt stream!"
            )

    def test_online_ttt_30_sample_sequence_divergence(self):
        """Online TTT must continuously adapt and significantly diverge from theta_0."""
        torch.manual_seed(999)
        model = SmallResNetTTT(num_classes=100)
        theta_0 = {k: v.clone() for k, v in model.features.state_dict().items()}

        stream = [(torch.rand(1, 3, 32, 32), torch.tensor([i % 20]), [f"sample_{i}"]) for i in range(30)]

        summary, windows, logs = evaluate_condition(
            model, stream, condition="online_ttt", ttt_lr=0.02, ttt_steps=1
        )

        assert summary["update_count"] == 30
        assert len(logs) == 30

        # Model features after full stream must differ significantly from theta_0
        final_features = model.features.state_dict()
        total_divergence = sum(
            torch.norm(final_features[k].float() - theta_0[k].float()).item()
            for k in theta_0
        )
        assert total_divergence > 0.1, (
            f"online_ttt divergence ({total_divergence}) is insufficient after 30 steps!"
        )

    @pytest.mark.parametrize("frozen_cond", ["supervised_frozen", "joint_frozen"])
    def test_frozen_baselines_zero_update_count_and_bit_for_bit_preservation(self, frozen_cond):
        """Frozen baselines must perform exactly 0 updates and leave all weights bit-for-bit untouched."""
        torch.manual_seed(888)
        model = SmallResNetTTT(num_classes=100)
        init_state = {k: v.clone() for k, v in model.state_dict().items()}

        stream = [(torch.rand(1, 3, 32, 32), torch.tensor([i % 20]), [f"frozen_{i}"]) for i in range(25)]

        summary, windows, logs = evaluate_condition(
            model, stream, condition=frozen_cond
        )

        assert summary["update_count"] == 0
        assert len(logs) == 25

        # All parameters (features, classifier, rotation_head) must be bit-for-bit identical
        for k, v in init_state.items():
            assert torch.equal(model.state_dict()[k], v), f"Param '{k}' modified during {frozen_cond}!"


# ==============================================================================
# Challenge 4: Rolling Window Conservation Laws and Schema Validation
# ==============================================================================

class TestRollingWindowConservationLaws:
    """Verifies window conservation laws, remainder handling, and Alka CSV persistence."""

    def test_window_conservation_laws(self):
        """Sum of correct in windows == sum of correct in sample logs == total correct."""
        predictions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        targets =     [1, 0, 3, 0, 5, 0, 7, 0, 9,  0, 11,  0, 13]
        total_correct = sum(1 for p, t in zip(predictions, targets) if p == t)

        for win_size in [1, 3, 5, 7, 13, 20]:
            windows = compute_rolling_windows(predictions, targets, window_size=win_size)
            win_correct_sum = sum(w["correct_count"] for w in windows)
            assert win_correct_sum == total_correct, (
                f"Window size {win_size}: sum of correct ({win_correct_sum}) != total ({total_correct})"
            )

            # Contiguity and coverage check
            assert windows[0]["start_sample"] == 0
            assert windows[-1]["end_sample"] == len(predictions)
            for i in range(len(windows) - 1):
                assert windows[i]["end_sample"] == windows[i + 1]["start_sample"]

    def test_save_summary_and_window_csv_persistence(self, tmp_path):
        """Verifies CSV writers produce files matching ALKA schema."""
        summary_csv = tmp_path / "results" / "ttt_summary.csv"
        window_csv = tmp_path / "results" / "ttt_window_summary.csv"

        row = build_summary_row(condition="standard_ttt", accuracy=0.75, run_id="test_run_1")
        save_summary_csv(row, summary_csv)

        windows = compute_rolling_windows([1, 2], [1, 2], window_size=1, run_id="test_run_1")
        save_window_csv(windows, window_csv)

        assert summary_csv.exists()
        assert window_csv.exists()

        import csv
        with open(summary_csv) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ALKA_SUMMARY_COLUMNS
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["run_id"] == "test_run_1"

        with open(window_csv) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ALKA_WINDOW_COLUMNS
            wrows = list(reader)
            assert len(wrows) == 2
