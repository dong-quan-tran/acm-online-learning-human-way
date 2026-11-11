"""Empirical Adversarial Stress Harness for Experiment B Test Suite.

Authored by Challenger R2-2.
Empirically tests the sensitivity of tests in tests/test_ttt_exp_b.py against
truncation mutants, momentum leakage, stream evaluation discrepancies, and
prediction audit violations.
"""

import copy
import inspect
import json
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.models as models
import src.data as data
import src.ttt_cifar100 as ttt
import tests.test_ttt_exp_b as target_test_module
from tests.test_ttt_exp_b import (
    TestTier1FeatureCoverage,
    TestTier1DataPipelineAndNoise,
    TestTier1AdaptationContract,
    TestTier2BoundaryCornerCases,
    TestTier3StateTransitions,
    TestTier4RealWorldScenarios,
    _unpack_eval_results,
)


class TestChallengerMutantSensitivity:
    """Stress tests verifying that target tests in test_ttt_exp_b.py CATCH mutants."""

    # --------------------------------------------------------------------------
    # CHALLENGE 1: Truncation Mutants vs test_unmasked_prediction_can_exceed_class_19
    # --------------------------------------------------------------------------

    def test_unmasked_catches_sliced_logits_mutant(self, monkeypatch):
        """Mutant: model.forward slices logits to 20 classes (logits[:, :20]).

        Target test MUST catch this mutant and raise an AssertionError.
        """
        orig_forward = models.SmallResNetTTT.forward

        def mutant_forward(self, x):
            logits = orig_forward(self, x)
            return logits[:, :20]  # Truncate to 0..19

        monkeypatch.setattr(models.SmallResNetTTT, "forward", mutant_forward)
        monkeypatch.setattr(target_test_module.SmallResNetTTT, "forward", mutant_forward)

        test_obj = TestTier2BoundaryCornerCases()
        with pytest.raises(AssertionError) as excinfo:
            test_obj.test_unmasked_prediction_can_exceed_class_19()
        assert "Expected 100-way logits" in str(excinfo.value) or "Argmax" in str(excinfo.value)

    def test_unmasked_catches_masked_logits_mutant(self, monkeypatch):
        """Mutant: model.forward keeps shape (1, 100) but zeros/masks indices 20..99.

        Target test MUST catch this mutant and raise an AssertionError.
        """
        orig_forward = models.SmallResNetTTT.forward

        def mutant_forward(self, x):
            logits = orig_forward(self, x)
            masked = logits.clone()
            masked[:, 20:] = -1e9
            return masked

        monkeypatch.setattr(models.SmallResNetTTT, "forward", mutant_forward)
        monkeypatch.setattr(target_test_module.SmallResNetTTT, "forward", mutant_forward)

        test_obj = TestTier2BoundaryCornerCases()
        with pytest.raises(AssertionError) as excinfo:
            test_obj.test_unmasked_prediction_can_exceed_class_19()
        assert "Real model forward pass must yield predicted class 55" in str(excinfo.value)

    def test_unmasked_catches_linear_head_20_classes_mutant(self, monkeypatch):
        """Mutant: classifier Linear layer is instantiated with 20 outputs instead of 100.

        Both test_unmasked_100_way_classifier and test_unmasked_prediction_can_exceed_class_19
        MUST catch this mutant.
        """
        orig_init = models.SmallResNetTTT.__init__

        def mutant_init(self, num_classes=100, **kwargs):
            orig_init(self, num_classes=20, **kwargs)

        monkeypatch.setattr(models.SmallResNetTTT, "__init__", mutant_init)
        monkeypatch.setattr(target_test_module.SmallResNetTTT, "__init__", mutant_init)

        tier1 = TestTier1FeatureCoverage()
        with pytest.raises(AssertionError):
            tier1.test_unmasked_100_way_classifier()

        tier2 = TestTier2BoundaryCornerCases()
        with pytest.raises((AssertionError, IndexError)):
            tier2.test_unmasked_prediction_can_exceed_class_19()

    # --------------------------------------------------------------------------
    # CHALLENGE 2: Optimizer & Momentum Leakage vs test_stateless_sgd_no_optimizer_leakage
    # --------------------------------------------------------------------------

    def test_stateless_sgd_catches_momentum_mutant(self, monkeypatch):
        """Mutant: adapt_and_predict_image uses momentum=0.9 instead of 0.0.

        test_stateless_sgd_no_optimizer_leakage MUST catch this mutant.
        """
        def mutant_adapt(model, image, ttt_lr=0.001, ttt_steps=1, lambda_rot=1.0):
            model.eval()
            for p in model.classifier.parameters():
                p.requires_grad = False
            for p in model.rotation_head.parameters():
                p.requires_grad = False
            for p in model.features.parameters():
                p.requires_grad = True

            rot_images, rot_labels = models.rotate_batch(image)
            pre_loss = 1.0
            if ttt_lr > 0.0 and ttt_steps > 0:
                opt = torch.optim.SGD(model.features.parameters(), lr=ttt_lr, momentum=0.9)
                for _ in range(ttt_steps):
                    opt.zero_grad()
                    rot_logits = model.forward_rotation(rot_images)
                    loss = lambda_rot * F.cross_entropy(rot_logits, rot_labels)
                    loss.backward()
                    opt.step()
            post_loss = 0.5
            logits = model(image)
            return logits, pre_loss, post_loss, 4

        monkeypatch.setattr(ttt, "adapt_and_predict_image", mutant_adapt)
        monkeypatch.setattr(target_test_module, "adapt_and_predict_image", mutant_adapt)

        test_obj = TestTier3StateTransitions()
        with pytest.raises(AssertionError) as excinfo:
            test_obj.test_stateless_sgd_no_optimizer_leakage(monkeypatch)
        assert "Khoi Check #7 violation" in str(excinfo.value)

    def test_stateless_sgd_catches_dirty_state_dict_mutant(self, monkeypatch):
        """Mutant: optimizer state is polluted with dirty buffers.

        test_stateless_sgd_no_optimizer_leakage MUST catch this mutant.
        """
        orig_sgd = torch.optim.SGD

        def mutant_sgd(*args, **kwargs):
            opt = orig_sgd(*args, **kwargs)
            orig_step = opt.step

            def dirty_step(*s_args, **s_kwargs):
                res = orig_step(*s_args, **s_kwargs)
                # Pollute opt.state with dirty buffer
                for p in opt.param_groups[0]["params"]:
                    opt.state[p]["dirty_momentum_buffer"] = torch.zeros_like(p)
                return res

            opt.step = dirty_step
            return opt

        monkeypatch.setattr(torch.optim, "SGD", mutant_sgd)

        test_obj = TestTier3StateTransitions()
        with pytest.raises(AssertionError) as excinfo:
            test_obj.test_stateless_sgd_no_optimizer_leakage(monkeypatch)
        assert "persistent state buffers" in str(excinfo.value) or "Khoi Check #7" in str(excinfo.value)

    def test_stateless_sgd_catches_all_parameters_optimized_mutant(self, monkeypatch):
        """Mutant: adapt_and_predict_image optimizes all model.parameters() instead of features only.

        test_stateless_sgd_no_optimizer_leakage MUST catch this mutant.
        """
        def mutant_adapt(model, image, ttt_lr=0.001, ttt_steps=1, lambda_rot=1.0):
            model.eval()
            rot_images, rot_labels = models.rotate_batch(image)
            # Incorrectly passes all model parameters to SGD
            opt = torch.optim.SGD(model.parameters(), lr=ttt_lr, momentum=0.0)
            if ttt_lr > 0.0 and ttt_steps > 0:
                for _ in range(ttt_steps):
                    opt.zero_grad()
                    rot_logits = model.forward_rotation(rot_images)
                    loss = lambda_rot * F.cross_entropy(rot_logits, rot_labels)
                    loss.backward()
                    opt.step()
            logits = model(image)
            return logits, 1.0, 0.5, 4

        monkeypatch.setattr(ttt, "adapt_and_predict_image", mutant_adapt)
        monkeypatch.setattr(target_test_module, "adapt_and_predict_image", mutant_adapt)

        test_obj = TestTier3StateTransitions()
        with pytest.raises(AssertionError) as excinfo:
            test_obj.test_stateless_sgd_no_optimizer_leakage(monkeypatch)
        assert "only optimize model.features" in str(excinfo.value)

    # --------------------------------------------------------------------------
    # CHALLENGE 3: Stream Adaptation Simulation vs test_stream_adaptation_simulation
    # --------------------------------------------------------------------------

    def test_stream_sim_catches_dropped_sample_mutant(self, monkeypatch):
        """Mutant: evaluate_condition silently drops 1 sample from logs.

        test_stream_adaptation_simulation MUST catch this mutant.
        """
        orig_eval = ttt.evaluate_condition

        def mutant_eval(*args, **kwargs):
            summary, windows, sample_logs = orig_eval(*args, **kwargs)
            return summary, windows, sample_logs[:-1]  # Drop last log

        monkeypatch.setattr(ttt, "evaluate_condition", mutant_eval)
        monkeypatch.setattr(target_test_module, "evaluate_condition", mutant_eval)

        test_obj = TestTier4RealWorldScenarios()
        with pytest.raises(AssertionError) as excinfo:
            test_obj.test_stream_adaptation_simulation()
        assert "recorded 9 predictions, expected 10" in str(excinfo.value)

    def test_stream_sim_catches_frozen_with_nonzero_updates_mutant(self, monkeypatch):
        """Mutant: supervised_frozen incorrectly reports updates > 0.

        test_stream_adaptation_simulation MUST catch this mutant.
        """
        orig_eval = ttt.evaluate_condition

        def mutant_eval(*args, **kwargs):
            summary, windows, sample_logs = orig_eval(*args, **kwargs)
            if kwargs.get("condition") == "supervised_frozen":
                summary["update_count"] = 5
            return summary, windows, sample_logs

        monkeypatch.setattr(ttt, "evaluate_condition", mutant_eval)
        monkeypatch.setattr(target_test_module, "evaluate_condition", mutant_eval)

        test_obj = TestTier4RealWorldScenarios()
        with pytest.raises(AssertionError) as excinfo:
            test_obj.test_stream_adaptation_simulation()
        assert "supervised_frozen must perform 0 updates" in str(excinfo.value)

    def test_stream_sim_catches_out_of_bounds_prediction_mutant(self, monkeypatch):
        """Mutant: model produces a prediction >= 100 (e.g. 105).

        test_stream_adaptation_simulation MUST catch this mutant.
        """
        orig_eval = ttt.evaluate_condition

        def mutant_eval(*args, **kwargs):
            summary, windows, sample_logs = orig_eval(*args, **kwargs)
            for log in sample_logs:
                log["predicted_label"] = 105
            return summary, windows, sample_logs

        monkeypatch.setattr(ttt, "evaluate_condition", mutant_eval)
        monkeypatch.setattr(target_test_module, "evaluate_condition", mutant_eval)

        test_obj = TestTier4RealWorldScenarios()
        with pytest.raises(AssertionError) as excinfo:
            test_obj.test_stream_adaptation_simulation()
        assert "outside unmasked [0..99] range" in str(excinfo.value)

    # --------------------------------------------------------------------------
    # CHALLENGE 4: Single Recorded Prediction vs test_single_recorded_prediction_per_image
    # --------------------------------------------------------------------------

    def test_single_pred_catches_duplicate_sample_id_mutant(self, monkeypatch):
        """Mutant: evaluate_condition produces duplicate sample_ids in logs.

        test_single_recorded_prediction_per_image MUST catch this mutant.
        """
        orig_eval = ttt.evaluate_condition

        def mutant_eval(*args, **kwargs):
            summary, windows, sample_logs = orig_eval(*args, **kwargs)
            if len(sample_logs) >= 2:
                sample_logs[1]["sample_id"] = sample_logs[0]["sample_id"]
            return summary, windows, sample_logs

        monkeypatch.setattr(ttt, "evaluate_condition", mutant_eval)
        monkeypatch.setattr(target_test_module, "evaluate_condition", mutant_eval)

        test_obj = TestTier4RealWorldScenarios()
        with pytest.raises(AssertionError) as excinfo:
            test_obj.test_single_recorded_prediction_per_image()
        assert "Duplicate sample IDs found in prediction log" in str(excinfo.value)

    def test_single_pred_catches_missing_sample_id_mutant(self, monkeypatch):
        """Mutant: evaluate_condition corrupts one sample_id to an unknown ID.

        test_single_recorded_prediction_per_image MUST catch this mutant.
        """
        orig_eval = ttt.evaluate_condition

        def mutant_eval(*args, **kwargs):
            summary, windows, sample_logs = orig_eval(*args, **kwargs)
            if len(sample_logs) >= 1:
                sample_logs[0]["sample_id"] = "unknown_phantom_id"
            return summary, windows, sample_logs

        monkeypatch.setattr(ttt, "evaluate_condition", mutant_eval)
        monkeypatch.setattr(target_test_module, "evaluate_condition", mutant_eval)

        test_obj = TestTier4RealWorldScenarios()
        with pytest.raises(AssertionError) as excinfo:
            test_obj.test_single_recorded_prediction_per_image()
        assert "Missing sample IDs in prediction log" in str(excinfo.value)
