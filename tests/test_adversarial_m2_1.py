"""Adversarial Empirical Stress-Testing Suite for Milestone 2 (Challenger 1).

Rigorous stress-testing of:
1. Weight resets vs carryover: standard_ttt exact bit-for-bit reset on sequences (15+ samples, lr=0.1, 1.0)
   vs online_ttt monotonic accumulation and divergence.
2. Label-free invariants: zero label leakage, zero label arguments in signature, unmasked logits shape (1, 100).
3. Head parameter immutability: classifier & rotation_head 100% frozen (0 epsilon change) under extreme lr & steps.
4. Stateless SGD: zero momentum/velocity/residual state leakage across adaptation steps.
5. State transition logic: evaluate_condition across conditions, loaders, edge cases, and failure modes.
"""

import copy
import inspect
import math
from typing import Any, Dict, List, Tuple
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import SmallResNetTTT, rotate_batch
from src.ttt_cifar100 import (
    ALKA_SUMMARY_COLUMNS,
    ALKA_WINDOW_COLUMNS,
    VALID_CONDITIONS,
    adapt_and_predict_image,
    compute_delta_accuracy,
    compute_rolling_windows,
    evaluate_condition,
)


# ==============================================================================
# Challenge 1: Weight Resets vs Carryover Dynamics
# ==============================================================================

class TestChallenge1WeightResetsVsCarryover:
    """Stress-tests standard_ttt episodic reset vs online_ttt state accumulation."""

    def test_standard_ttt_15_step_sequence_bit_for_bit_reset(self):
        """Verifies standard_ttt feature reset on a 15-sample sequence under high learning rate."""
        torch.manual_seed(42)
        model = SmallResNetTTT(num_classes=100)
        theta_0 = {k: v.clone() for k, v in model.features.state_dict().items()}

        # 15 distinct images
        images = [torch.rand(1, 3, 32, 32) for _ in range(15)]
        ttt_lr = 0.05
        ttt_steps = 2

        for step_idx, img in enumerate(images):
            # Pre-condition check: model features must be IDENTICAL to theta_0 before adaptation
            curr_features = model.features.state_dict()
            for k, v in theta_0.items():
                assert torch.equal(curr_features[k], v), (
                    f"Step {step_idx}: Feature param '{k}' diverged from theta_0 BEFORE adaptation!"
                )

            # Adapt and predict
            seq_logits, pre_loss, post_loss, _ = adapt_and_predict_image(
                model, img, ttt_lr=ttt_lr, ttt_steps=ttt_steps
            )

            # Features DID adapt
            adapted_features = {k: v.clone() for k, v in model.features.state_dict().items()}
            assert any(not torch.equal(adapted_features[k], theta_0[k]) for k in theta_0), (
                f"Step {step_idx}: Adaptation failed to modify features with lr={ttt_lr}"
            )

            # Reset model features to theta_0 (standard_ttt invariant)
            model.features.load_state_dict(theta_0)

            # Post-reset check: verify bit-for-bit restoration
            for k, v in theta_0.items():
                assert torch.equal(model.features.state_dict()[k], v), (
                    f"Step {step_idx}: Feature param '{k}' failed bit-for-bit reset to theta_0!"
                )

            # Isolated baseline comparison: evaluate the same image from fresh model
            fresh_model = SmallResNetTTT(num_classes=100)
            fresh_model.load_state_dict(model.state_dict())
            iso_logits, iso_pre_loss, iso_post_loss, _ = adapt_and_predict_image(
                fresh_model, img, ttt_lr=ttt_lr, ttt_steps=ttt_steps
            )

            # Sequence prediction and isolated prediction must be strictly bit-for-bit identical
            assert torch.equal(seq_logits, iso_logits), (
                f"Step {step_idx}: Logits from sequence ({seq_logits}) != isolated ({iso_logits})"
            )
            assert pre_loss == pytest.approx(iso_pre_loss, abs=1e-6)
            assert post_loss == pytest.approx(iso_post_loss, abs=1e-6)

    def test_online_ttt_15_step_sequence_monotonic_accumulation(self):
        """Verifies online_ttt strictly accumulates updates and diverges from standard_ttt and theta_0."""
        torch.manual_seed(123)
        model_online = SmallResNetTTT(num_classes=100)
        theta_0 = {k: v.clone() for k, v in model_online.features.state_dict().items()}

        images = [torch.rand(1, 3, 32, 32) for _ in range(15)]
        ttt_lr = 0.02
        ttt_steps = 1

        prev_features = {k: v.clone() for k, v in theta_0.items()}
        l2_distances_from_theta_0 = []

        for step_idx, img in enumerate(images):
            adapt_and_predict_image(model_online, img, ttt_lr=ttt_lr, ttt_steps=ttt_steps)
            curr_features = {k: v.clone() for k, v in model_online.features.state_dict().items()}

            # Invariant 1: Step must alter weights relative to previous step
            assert any(not torch.equal(curr_features[k], prev_features[k]) for k in theta_0), (
                f"Step {step_idx}: online_ttt did not change feature weights"
            )

            # Compute L2 distance from theta_0
            dist = 0.0
            for k in theta_0:
                dist += torch.norm(curr_features[k].float() - theta_0[k].float()).item() ** 2
            l2_dist = math.sqrt(dist)
            l2_distances_from_theta_0.append(l2_dist)

            prev_features = curr_features

        # Invariant 2: Total divergence from theta_0 is strictly positive
        assert l2_distances_from_theta_0[-1] > 0.05, (
            f"online_ttt failed to meaningfully diverge from theta_0 after 15 steps: dist={l2_distances_from_theta_0[-1]}"
        )

        # Invariant 3: Comparison against standard TTT on the final sample (step 14)
        model_std = SmallResNetTTT(num_classes=100)
        model_std.features.load_state_dict(theta_0)
        std_logits, _, _, _ = adapt_and_predict_image(model_std, images[-1], ttt_lr=ttt_lr, ttt_steps=ttt_steps)
        online_logits = model_online(images[-1])

        # Online predictions and features must diverge from standard TTT
        diverged_from_std = any(
            not torch.equal(model_online.features.state_dict()[k], model_std.features.state_dict()[k])
            for k in theta_0
        )
        assert diverged_from_std, (
            "online_ttt feature weights unexpectedly equal standard_ttt weights at step 15"
        )
        assert not torch.equal(online_logits, std_logits), (
            "online_ttt logits unexpectedly equal standard_ttt logits at step 15"
        )


# ==============================================================================
# Challenge 2: Label-Free Invariants & Zero Information Leakage
# ==============================================================================

class TestChallenge2LabelFreeInvariants:
    """Verifies adapt_and_predict_image operates strictly without label access or leakage."""

    def test_signature_label_free_strict_allowlist(self):
        """Inspects signature: parameter names must NOT contain any label/target/class substrings."""
        sig = inspect.signature(adapt_and_predict_image)
        forbidden_substrings = ["label", "target", "cls", "ground_truth", "y", "class_id"]

        for name in sig.parameters:
            for forbidden in forbidden_substrings:
                assert forbidden not in name.lower(), (
                    f"Label leakage risk: Parameter '{name}' contains forbidden substring '{forbidden}'"
                )

        # Verify reject unexpected keyword arguments
        model = SmallResNetTTT(num_classes=100)
        img = torch.rand(1, 3, 32, 32)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            adapt_and_predict_image(model, img, labels=torch.tensor([5]))

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            adapt_and_predict_image(model, img, y=5)

    def test_logits_shape_and_range_unmasked(self):
        """Verifies logits tensor is strictly (1, 100) unmasked across all CIFAR-100 classes."""
        model = SmallResNetTTT(num_classes=100)
        img_3d = torch.rand(3, 32, 32)
        img_4d = torch.rand(1, 3, 32, 32)

        out_3d = adapt_and_predict_image(model, img_3d, ttt_lr=0.001)
        assert out_3d[0].shape == (1, 100), f"Expected (1, 100) for 3D input, got {out_3d[0].shape}"

        out_4d = adapt_and_predict_image(model, img_4d, ttt_lr=0.001)
        assert out_4d[0].shape == (1, 100), f"Expected (1, 100) for 4D input, got {out_4d[0].shape}"

    def test_invalid_tensor_ranks_rejected(self):
        """Verifies that 2D or 5D or multi-image tensors are rejected."""
        model = SmallResNetTTT(num_classes=100)

        # 2D tensor
        with pytest.raises(ValueError, match="expects 3D or.*4D"):
            adapt_and_predict_image(model, torch.rand(32, 32))

        # 5D tensor
        with pytest.raises(ValueError, match="expects 3D or.*4D"):
            adapt_and_predict_image(model, torch.rand(1, 1, 3, 32, 32))

        # 4D tensor with batch size > 1
        with pytest.raises(ValueError, match="expects 3D or.*4D"):
            adapt_and_predict_image(model, torch.rand(2, 3, 32, 32))


# ==============================================================================
# Challenge 3: Head Parameter Immutability Under Adversarial LR & Steps
# ==============================================================================

class TestChallenge3HeadParameterImmutability:
    """Stress-tests classification and rotation head parameter immutability."""

    @pytest.mark.parametrize("extreme_lr", [0.0, 0.1, 1.0, 10.0])
    @pytest.mark.parametrize("steps", [1, 5, 10])
    def test_head_parameters_strictly_identical_zero_epsilon(self, extreme_lr, steps):
        """Under high learning rates and multiple steps, heads must not change by 1 epsilon."""
        model = SmallResNetTTT(num_classes=100)

        # Snapshot heads and normalization buffers
        cls_weights_before = {k: v.clone() for k, v in model.classifier.state_dict().items()}
        rot_weights_before = {k: v.clone() for k, v in model.rotation_head.state_dict().items()}
        mean_before = model.mean.clone()
        std_before = model.std.clone()

        img = torch.rand(1, 3, 32, 32)
        adapt_and_predict_image(model, img, ttt_lr=extreme_lr, ttt_steps=steps)

        # Classifier head exact tensor equality
        cls_weights_after = model.classifier.state_dict()
        for k in cls_weights_before:
            diff = (cls_weights_after[k] - cls_weights_before[k]).abs().max().item()
            assert diff == 0.0, (
                f"Classifier param '{k}' mutated! Max diff: {diff} (lr={extreme_lr}, steps={steps})"
            )
            assert torch.equal(cls_weights_after[k], cls_weights_before[k])

        # Rotation head exact tensor equality
        rot_weights_after = model.rotation_head.state_dict()
        for k in rot_weights_before:
            diff = (rot_weights_after[k] - rot_weights_before[k]).abs().max().item()
            assert diff == 0.0, (
                f"Rotation head param '{k}' mutated! Max diff: {diff} (lr={extreme_lr}, steps={steps})"
            )
            assert torch.equal(rot_weights_after[k], rot_weights_before[k])

        # Normalization buffers
        assert torch.equal(model.mean, mean_before)
        assert torch.equal(model.std, std_before)

        # Autograd gradients must be None and requires_grad False
        for name, p in model.classifier.named_parameters():
            assert not p.requires_grad, f"Classifier parameter {name} has requires_grad=True!"
            assert p.grad is None, f"Classifier parameter {name} has non-None grad!"

        for name, p in model.rotation_head.named_parameters():
            assert not p.requires_grad, f"Rotation head parameter {name} has requires_grad=True!"
            assert p.grad is None, f"Rotation head parameter {name} has non-None grad!"


# ==============================================================================
# Challenge 4: Stateless SGD & Zero Optimizer Leakage
# ==============================================================================

class TestChallenge4StatelessSGD:
    """Verifies that no optimizer momentum, velocity, or residual state leaks."""

    def test_optimizer_statelessness_and_buffer_inspection(self, monkeypatch):
        """Monitors SGD optimizer instantiations across sequential adaptation steps."""
        model = SmallResNetTTT(num_classes=100)
        img = torch.rand(1, 3, 32, 32)

        instantiated_optimizers: List[torch.optim.SGD] = []
        real_sgd = torch.optim.SGD

        def spy_sgd(params, *args, **kwargs):
            opt = real_sgd(params, *args, **kwargs)
            instantiated_optimizers.append(opt)
            return opt

        monkeypatch.setattr(torch.optim, "SGD", spy_sgd)

        # Run 5 sequential adaptation steps
        for step in range(5):
            adapt_and_predict_image(model, img, ttt_lr=0.01, ttt_steps=2)

        assert len(instantiated_optimizers) == 5, (
            f"Expected 5 separate SGD instances, got {len(instantiated_optimizers)}"
        )

        # Verify all instances are distinct objects
        opt_ids = [id(opt) for opt in instantiated_optimizers]
        assert len(set(opt_ids)) == 5, "Optimizer instance was reused across adaptation calls!"

        for idx, opt in enumerate(instantiated_optimizers):
            # Check momentum is strictly 0.0
            momentum = opt.param_groups[0].get("momentum", 0.0)
            assert momentum == 0.0, f"Opt {idx}: momentum is {momentum}, expected 0.0"

            # Check dampening is 0.0 and nesterov is False
            assert opt.param_groups[0].get("dampening", 0.0) == 0.0
            assert opt.param_groups[0].get("nesterov", False) is False

            # Check optimizer state dictionary is empty
            assert len(opt.state) == 0, (
                f"Opt {idx}: optimizer state dict contains {len(opt.state)} entries!"
            )

        # Check feature parameters have no leftover gradients
        for name, p in model.features.named_parameters():
            assert p.grad is None, (
                f"Feature param '{name}' has non-None gradient after adapt_and_predict_image"
            )


# ==============================================================================
# Challenge 5: State Transition Logic (evaluate_condition) & Error Paths
# ==============================================================================

class TestChallenge5StateTransitionsAndEdgeCases:
    """Probes evaluate_condition transitions, input structures, and failure modes."""

    def test_invalid_conditions_rejected(self):
        """Verifies evaluate_condition raises ValueError on invalid condition strings."""
        model = SmallResNetTTT(num_classes=100)
        dummy_loader = [(torch.rand(1, 3, 32, 32), torch.tensor([1]), ["sid_0"])]

        for invalid_cond in ["online", "standard", "test_time", "random", "", None]:
            with pytest.raises((ValueError, TypeError)):
                evaluate_condition(model, dummy_loader, condition=invalid_cond)

    def test_negative_lr_and_steps_rejected(self):
        """Verifies negative lr and steps are rejected."""
        model = SmallResNetTTT(num_classes=100)
        img = torch.rand(1, 3, 32, 32)

        with pytest.raises(ValueError, match="ttt_lr"):
            adapt_and_predict_image(model, img, ttt_lr=-0.01)

        with pytest.raises(ValueError, match="ttt_steps"):
            adapt_and_predict_image(model, img, ttt_steps=-1)

    def test_standard_dataloader_batch_format(self):
        """Tests evaluate_condition with standard PyTorch DataLoader batch (images, tensor_labels, sids)."""
        model = SmallResNetTTT(num_classes=100)
        # Tensor labels (standard DataLoader collation)
        stream = [
            (torch.rand(1, 3, 32, 32), torch.tensor([5]), ["test:0"]),
            (torch.rand(1, 3, 32, 32), torch.tensor([12]), ["test:1"]),
        ]

        summary, windows, logs = evaluate_condition(
            model, stream, condition="standard_ttt", ttt_lr=0.001
        )
        assert summary["update_count"] == 2
        assert len(logs) == 2
        assert logs[0]["sample_id"] == "test:0"
        assert logs[1]["sample_id"] == "test:1"

    def test_scalar_int_label_loader_resolution(self):
        """Empirically verifies that loader yielding scalar int labels does not crash evaluate_condition."""
        model = SmallResNetTTT(num_classes=100)
        # Stream with scalar int label and string sample id
        stream_with_int_label = [
            (torch.rand(1, 3, 32, 32), 5, "test:101"),
            (torch.rand(1, 3, 32, 32), 12, "test:102"),
        ]

        summary, windows, logs = evaluate_condition(
            model, stream_with_int_label, condition="standard_ttt", ttt_lr=0.001
        )
        assert summary["update_count"] == 2
        assert len(logs) == 2
        assert logs[0]["true_label"] == 5
        assert logs[0]["sample_id"] == "test:101"
        assert logs[1]["true_label"] == 12
        assert logs[1]["sample_id"] == "test:102"
        assert "predicted_class" in logs[0]
        assert "is_correct" in logs[0]
