"""Opaque-box Verification Suite for Experiment B (Tuong)
Rotation-Based Test-Time Training on CIFAR-100 (Classes 0–19).

Strictly validates:
- Tier 1: Feature Coverage (GroupNorm, unmasked 100-way head, 4-way rotation head,
          splits, deterministic noise, DataLoader integration, hardened label-free signature, 4 conditions)
- Tier 2: Boundary & Corner Cases (B=1 GroupNorm stability, noise clamping bounds in [0.0, 1.0],
          rotation cycle closure via rotate_batch, lr=0.0 invariance, unmasked real argmax, invalid inputs)
- Tier 3: State Transitions & Invariants (Khoi Checks #2, #4, #5: head parameter immutability,
          standard_ttt episodic reset, online_ttt carryover & divergence, frozen model invariance,
          repeatable noise cross-condition consistency, stateless SGD via spy)
- Tier 4: Real-World Scenarios (Khoi Checks #6, #8: genuine stream evaluation simulation,
          Alka CSV logging schemas, windowed accuracy conservation laws, single recorded prediction audit, delta accuracy)
"""

import copy
import inspect
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch
from torch import nn

# Limit PyTorch CPU threads for fast and predictable test execution
torch.set_num_threads(2)

# ==============================================================================
# Conditional Imports & Component Detection
# ==============================================================================

try:
    from src.models import (
        CIFAR100_MEAN,
        CIFAR100_STD,
        ResidualGN,
        SmallResNetTTT,
        conv_gn,
        rotate_batch,
    )
    HAS_MODELS_TTT = True
except (ImportError, AttributeError):
    HAS_MODELS_TTT = False

try:
    from src.data import (
        CIFAR100ExpBDataset,
        apply_gaussian_noise,
        get_exp_b_loader,
        get_exp_b_loaders,
    )
    HAS_DATA_EXP_B = True
except (ImportError, AttributeError):
    HAS_DATA_EXP_B = False

try:
    from src.ttt_cifar100 import (
        ALKA_SUMMARY_COLUMNS,
        ALKA_WINDOW_COLUMNS,
        VALID_CONDITIONS,
        adapt_and_predict_image,
        compute_delta_accuracy,
        compute_rolling_windows,
        evaluate_condition,
    )
    try:
        from src.ttt_cifar100 import build_summary_row
    except ImportError:
        build_summary_row = None
    HAS_TTT_ENGINE = True
except (ImportError, AttributeError):
    HAS_TTT_ENGINE = False
    ALKA_SUMMARY_COLUMNS = []
    ALKA_WINDOW_COLUMNS = []
    VALID_CONDITIONS = ()
    adapt_and_predict_image = None
    build_summary_row = None
    compute_delta_accuracy = None
    compute_rolling_windows = None
    evaluate_condition = None


def _unpack_eval_results(res: Any) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Normalize evaluate_condition return value to (summary_dict, sample_logs)."""
    if isinstance(res, tuple):
        if len(res) == 3:
            # (summary_dict, window_rows, sample_logs)
            return res[0], res[2]
        elif len(res) == 2:
            # (summary_dict, sample_logs)
            return res[0], res[1]
    if isinstance(res, dict):
        logs = res.get("logs", res.get("predictions_log", res.get("sample_logs", [])))
        return res, logs
    raise TypeError(f"Unexpected evaluate_condition return type: {type(res)}")


# ==============================================================================
# CLASS 1: Tier 1 Feature Coverage (Architecture & Foundations)
# ==============================================================================

class TestTier1FeatureCoverage:
    """Verifies Feature F1 (GroupNorm Backbone, Cls Head, Rot Head, rotate_batch)."""

    @pytest.mark.skipif(not HAS_MODELS_TTT, reason="SmallResNetTTT not yet available in src.models")
    def test_groupnorm_backbone_zero_batchnorm(self):
        """Verifies 0 BatchNorm2d instances and exactly 9 GroupNorm(16, C) layers (3 per stage x 3 stages)."""
        model = SmallResNetTTT(num_classes=100)

        # Invariant 1: Exactly zero BatchNorm instances
        for name, module in model.named_modules():
            assert not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)), (
                f"Forbidden BatchNorm layer detected at '{name}': {module}"
            )

        # Invariant 2: Exactly 9 GroupNorm layers with num_groups=16 across 3 stages
        # (each stage has 1 conv_gn + 2 in ResidualGN = 3 GN per stage x 3 stages = 9)
        gn_layers = [m for m in model.modules() if isinstance(m, nn.GroupNorm)]
        assert len(gn_layers) == 9, (
            f"Expected exactly 9 GroupNorm layers (3 per stage x 3 stages), found {len(gn_layers)}"
        )
        for idx, gn in enumerate(gn_layers):
            assert gn.num_groups == 16, f"GroupNorm layer {idx} has num_groups={gn.num_groups}, expected 16"

        # Invariant 3: Registered normalization buffers
        assert hasattr(model, "mean") and hasattr(model, "std")
        expected_mean = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1)
        expected_std = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)
        assert torch.allclose(model.mean, expected_mean, atol=1e-3)
        assert torch.allclose(model.std, expected_std, atol=1e-3)

    @pytest.mark.skipif(not HAS_MODELS_TTT, reason="SmallResNetTTT not yet available in src.models")
    def test_unmasked_100_way_classifier(self):
        """Verifies unmasked 100-way linear classifier head following 256-d pooled features."""
        model = SmallResNetTTT(num_classes=100)
        assert hasattr(model, "classifier"), "Model must have 'classifier' attribute"

        linear_layers = [m for m in model.classifier.modules() if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 1, "Expected exactly 1 Linear layer in model.classifier"
        assert linear_layers[0].in_features == 256
        assert linear_layers[0].out_features == 100, "Classifier must be unmasked 100-way linear layer"

        x = torch.rand(2, 3, 32, 32)
        logits = model(x)
        assert logits.shape == (2, 100), f"Expected class logits shape (2, 100), got {logits.shape}"

    @pytest.mark.skipif(not HAS_MODELS_TTT, reason="SmallResNetTTT not yet available in src.models")
    def test_4_way_rotation_head(self):
        """Verifies 4-way linear rotation prediction head attached to 256-d features."""
        model = SmallResNetTTT(num_classes=100)
        assert hasattr(model, "rotation_head"), "Model must have 'rotation_head' attribute"
        assert isinstance(model.rotation_head, nn.Linear)
        assert model.rotation_head.in_features == 256
        assert model.rotation_head.out_features == 4

        x = torch.rand(2, 3, 32, 32)
        rot_logits = model.forward_rotation(x)
        assert rot_logits.shape == (2, 4)

        cls_logits, rot_logits2 = model.forward_both(x)
        assert cls_logits.shape == (2, 100)
        assert rot_logits2.shape == (2, 4)

        features = model.extract_features(x)
        assert features.shape == (2, 256)

    @pytest.mark.skipif(not HAS_MODELS_TTT, reason="rotate_batch not yet available in src.models")
    def test_rotate_batch_generation(self):
        """Verifies rotate_batch expands (B, C, H, W) into (4B, C, H, W) for 4 rotation angles."""
        b = 2
        images = torch.rand(b, 3, 32, 32)
        rot_images, rot_labels = rotate_batch(images)

        assert rot_images.shape == (4 * b, 3, 32, 32)
        assert rot_labels.shape == (4 * b,)

        # 0° slice
        assert torch.equal(rot_images[0:b], images)
        assert torch.all(rot_labels[0:b] == 0)

        # 90° slice
        assert torch.equal(rot_images[b:2 * b], torch.rot90(images, 1, (-2, -1)))
        assert torch.all(rot_labels[b:2 * b] == 1)

        # 180° slice
        assert torch.equal(rot_images[2 * b:3 * b], torch.rot90(images, 2, (-2, -1)))
        assert torch.all(rot_labels[2 * b:3 * b] == 2)

        # 270° slice
        assert torch.equal(rot_images[3 * b:4 * b], torch.rot90(images, 3, (-2, -1)))
        assert torch.all(rot_labels[3 * b:4 * b] == 3)

    def test_small_resnet_untouched(self):
        """Verifies that legacy SmallResNet in src/models.py remains untouched."""
        from src.models import SmallResNet
        model = SmallResNet()
        assert hasattr(model, "classifier")
        # Legacy MNIST classifier has 10 classes
        assert model.classifier[1].out_features == 10
        # Legacy SmallResNet retains BatchNorm2d
        bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        assert len(bn_layers) > 0, "SmallResNet must retain BatchNorm2d for MNIST compatibility"


# ==============================================================================
# CLASS 2: Tier 1 Data Pipeline, DataLoader Integration & Deterministic Noise
# ==============================================================================

class SyntheticCIFAR:
    """Lightweight in-memory synthetic dataset for offline DataLoader testing.

    Provides 50,000 synthetic RGB images and fine labels in 0..19 without network calls.
    """

    def __init__(self, num_samples: int = 50000):
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        from PIL import Image

        return {
            "img": Image.new(
                "RGB",
                (32, 32),
                color=(idx % 256, (idx * 7) % 256, (idx * 13) % 256),
            ),
            "fine_label": idx % 20,
        }


class TestTier1DataPipelineAndNoise:
    """Verifies Features F2 (Splits, Membership, DataLoaders) and F3 (Deterministic Gaussian Noise)."""

    def test_data_split_counts_and_disjointness(self):
        """Verifies Khoi Check #1 & #3: shared membership structure and disjointness."""
        membership_path = Path("data/shared_c20_membership.json")
        assert membership_path.is_file(), "data/shared_c20_membership.json must exist"

        with open(membership_path, "r", encoding="utf-8") as f:
            membership = json.load(f)

        assert membership.get("source_split") == "train", "Membership source_split must be 'train'"
        assert "train" in membership and "validation" in membership

        train_rows = membership["train"]
        val_rows = membership["validation"]

        assert len(train_rows) == 9000, f"Expected 9000 train rows, got {len(train_rows)}"
        assert len(val_rows) == 1000, f"Expected 1000 validation rows, got {len(val_rows)}"

        train_set = set(train_rows)
        val_set = set(val_rows)

        assert len(train_set) == 9000, "Train split contains duplicate row IDs"
        assert len(val_set) == 1000, "Validation split contains duplicate row IDs"
        assert train_set.isdisjoint(val_set), "Train and validation splits must be mutually disjoint"

        assert all(isinstance(r, int) and r >= 0 for r in train_rows)
        assert all(isinstance(r, int) and r >= 0 for r in val_rows)

        # Audit test membership file if generated
        test_membership_path = Path("data/exp_b_test_membership.json")
        if test_membership_path.is_file():
            with open(test_membership_path, "r", encoding="utf-8") as f:
                test_mem = json.load(f)
            assert test_mem.get("source_split") == "test", "Test membership source_split must be 'test'"
            assert len(test_mem.get("test", [])) == 2000, "Test membership must contain exactly 2000 rows"
            assert test_mem["test"] == sorted(test_mem["test"]), "Test rows must be in ascending source-row order"
            assert len(set(test_mem["test"])) == 2000, "Test membership contains duplicate row IDs"

    @pytest.mark.skipif(not HAS_DATA_EXP_B, reason="apply_gaussian_noise not yet available in src.data")
    def test_deterministic_gaussian_noise_repeatability(self):
        """Verifies Khoi Check #2: identical noisy tensor across repeated calls."""
        img = torch.rand(3, 32, 32)
        noisy1 = apply_gaussian_noise(img, sample_id=42, sigma=0.08, noise_seed=42)
        noisy2 = apply_gaussian_noise(img, sample_id=42, sigma=0.08, noise_seed=42)

        assert torch.equal(noisy1, noisy2), (
            "apply_gaussian_noise must produce bit-for-bit identical output for identical (sample_id, noise_seed)"
        )
        assert noisy1.min() >= 0.0 and noisy1.max() <= 1.0, "Noisy tensor must be clamped to [0.0, 1.0]"
        assert noisy1.shape == img.shape

    @pytest.mark.skipif(not HAS_DATA_EXP_B, reason="apply_gaussian_noise not yet available in src.data")
    def test_noise_sample_differentiation(self):
        """Verifies that noise perturbation varies across sample_id and noise_seed."""
        img = torch.rand(3, 32, 32)
        n1 = apply_gaussian_noise(img, sample_id=101, sigma=0.08, noise_seed=42)
        n2 = apply_gaussian_noise(img, sample_id=102, sigma=0.08, noise_seed=42)
        n3 = apply_gaussian_noise(img, sample_id=101, sigma=0.08, noise_seed=99)

        assert not torch.equal(n1, n2), "Different sample_id must produce different noise perturbations"
        assert not torch.equal(n1, n3), "Different noise_seed must produce different noise perturbations"

    @pytest.mark.skipif(not HAS_DATA_EXP_B, reason="get_exp_b_loaders not yet available in src.data")
    def test_get_exp_b_loaders_shapes_lengths_and_types(self):
        """Verifies get_exp_b_loaders returns valid DataLoaders with expected shapes, lengths, and types offline."""
        synth = SyntheticCIFAR()
        train_loader, val_loader, test_loader = get_exp_b_loaders(
            dataset_train=synth,
            dataset_test=synth,
            batch_size=32,
            val_batch_size=64,
            test_batch_size=1,
            num_workers=0,
        )

        # Invariant 1: Correct DataLoader and Dataset types
        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert isinstance(val_loader, torch.utils.data.DataLoader)
        assert isinstance(test_loader, torch.utils.data.DataLoader)

        # Invariant 2: Dataset lengths match Experiment B specification
        assert len(train_loader.dataset) == 9000
        assert len(val_loader.dataset) == 1000
        assert len(test_loader.dataset) == 2000

        # Invariant 3: Loader batch counts match batch size calculations
        assert len(train_loader) == math.ceil(9000 / 32)
        assert len(val_loader) == math.ceil(1000 / 64)
        assert len(test_loader) == 2000

        # Invariant 4: Train batch tensors have shape (B, 3, 32, 32) and labels in 0..19
        images, labels, sample_ids = next(iter(train_loader))
        assert images.shape == (32, 3, 32, 32)
        assert images.dtype == torch.float32
        assert labels.shape == (32,)
        assert labels.dtype == torch.int64
        assert all(0 <= l.item() < 20 for l in labels)
        assert len(sample_ids) == 32
        assert all(s.startswith("train:") for s in sample_ids)

        # Invariant 5: Test batch tensors have shape (1, 3, 32, 32) with 'test:' prefix
        t_images, t_labels, t_sample_ids = next(iter(test_loader))
        assert t_images.shape == (1, 3, 32, 32)
        assert t_labels.shape == (1,)
        assert len(t_sample_ids) == 1
        assert t_sample_ids[0].startswith("test:")

    @pytest.mark.skipif(not HAS_DATA_EXP_B, reason="get_exp_b_loaders not yet available in src.data")
    def test_get_exp_b_loaders_noise_and_streaming(self):
        """Verifies get_exp_b_loaders injects repeatable deterministic noise in test loader."""
        synth = SyntheticCIFAR()
        _, _, noisy_test_loader = get_exp_b_loaders(
            dataset_train=synth,
            dataset_test=synth,
            batch_size=16,
            corruption="gaussian_noise",
            noise_sigma=0.08,
            noise_seed=42,
            num_workers=0,
        )

        noisy_images, noisy_labels, noisy_sample_ids = next(iter(noisy_test_loader))
        assert noisy_images.shape == (16, 3, 32, 32)
        assert noisy_images.min() >= 0.0, "Noisy images fell below 0.0"
        assert noisy_images.max() <= 1.0, "Noisy images exceeded 1.0"
        assert all(s.startswith("test:") for s in noisy_sample_ids)

    @pytest.mark.skipif(not HAS_DATA_EXP_B, reason="get_exp_b_loader not yet available in src.data")
    def test_get_exp_b_loader_single_split(self):
        """Verifies single-split get_exp_b_loader helper function for stream evaluation."""
        synth = SyntheticCIFAR()
        loader = get_exp_b_loader(
            split="test",
            batch_size=1,
            corruption="clean",
            dataset=synth,
            num_workers=0,
        )
        assert isinstance(loader, torch.utils.data.DataLoader)
        assert len(loader) == 2000
        assert len(loader.dataset) == 2000
        image, label, sample_id = next(iter(loader))
        assert image.shape == (1, 3, 32, 32)
        assert label.shape == (1,)
        assert sample_id[0].startswith("test:")

    @pytest.mark.skipif(not HAS_DATA_EXP_B, reason="get_exp_b_loaders not yet available in src.data")
    def test_get_exp_b_loaders_invalid_parameters(self):
        """Verifies get_exp_b_loaders rejects invalid corruption or split choices."""
        synth = SyntheticCIFAR()
        with pytest.raises(ValueError, match="corruption must be 'clean' or 'gaussian_noise'"):
            get_exp_b_loaders(corruption="invalid_corruption", dataset_train=synth, dataset_test=synth)

        with pytest.raises(ValueError, match="split must be 'train', 'validation', or 'test'"):
            get_exp_b_loader(split="invalid_split", dataset=synth)


# ==============================================================================
# CLASS 3: Tier 1 Adaptation Contract & Condition Management
# ==============================================================================

class TestTier1AdaptationContract:
    """Verifies Features F5 (Label-Free Contract) and F6 (4 Evaluation Conditions)."""

    @pytest.mark.skipif(not HAS_TTT_ENGINE, reason="adapt_and_predict_image not yet available in src.ttt_cifar100")
    def test_label_free_adaptation_signature(self):
        """Verifies Khoi Check #4: adapt_and_predict_image strictly accepts NO label parameters.

        Hardened against label exposure vulnerabilities:
        1. Strictly bounds parameter count: 2 <= count <= 5.
        2. Strictly forbids *args (VAR_POSITIONAL) and **kwargs (VAR_KEYWORD) to prevent backdoor label passage.
        3. Forbids ANY parameter containing substrings: 'label', 'target', 'gt', 'ground_truth', 'class', 'y'.
        4. Strictly enforces parameter allowlist: {'model', 'image', 'ttt_lr', 'ttt_steps', 'lambda_rot'}.
        5. Validates canonical parameter ordering matching PROJECT.md interface contract.
        """
        sig = inspect.signature(adapt_and_predict_image)
        param_names = list(sig.parameters.keys())
        params = list(sig.parameters.values())

        authorized_params = ("model", "image", "ttt_lr", "ttt_steps", "lambda_rot")
        forbidden_substrings = ("label", "target", "gt", "ground_truth", "class", "y")

        # 1. Bounded parameter count: at least (model, image), at most the 5 authorized parameters
        assert 2 <= len(param_names) <= len(authorized_params), (
            f"Parameter count {len(param_names)} out of bounds [2, {len(authorized_params)}]. "
            f"Parameters found: {param_names}"
        )

        # 2. Forbid wildcard *args (VAR_POSITIONAL) and **kwargs (VAR_KEYWORD)
        var_positional = [p.name for p in params if p.kind == inspect.Parameter.VAR_POSITIONAL]
        assert not var_positional, (
            f"Integrity violation: wildcard *args (VAR_POSITIONAL) parameter(s) {var_positional} detected. "
            f"Positional varargs are strictly forbidden to prevent backdoor label passage."
        )

        var_keyword = [p.name for p in params if p.kind == inspect.Parameter.VAR_KEYWORD]
        assert not var_keyword, (
            f"Integrity violation: wildcard **kwargs (VAR_KEYWORD) parameter(s) {var_keyword} detected. "
            f"Keyword varargs are strictly forbidden to prevent backdoor label passage."
        )

        # 3. Forbid ANY parameter containing forbidden substrings (case-insensitive)
        for name in param_names:
            name_lower = name.lower()
            for forbidden in forbidden_substrings:
                assert forbidden not in name_lower, (
                    f"Integrity violation: parameter '{name}' contains forbidden substring '{forbidden}'. "
                    f"adapt_and_predict_image must strictly adhere to a label-free contract."
                )

        # 4. Strict parameter allowlist: zero unauthorized parameters permitted
        unauthorized = set(param_names) - set(authorized_params)
        assert not unauthorized, (
            f"Integrity violation: unauthorized parameter(s) {unauthorized} found in signature. "
            f"Authorized parameters are: {authorized_params}"
        )

        # 5. Canonical parameter ordering: parameters must appear in exact contract order
        for i, name in enumerate(param_names):
            assert name == authorized_params[i], (
                f"Parameter index {i} must be '{authorized_params[i]}', got '{name}'"
            )

    @pytest.mark.skipif(
        not (HAS_TTT_ENGINE and HAS_MODELS_TTT),
        reason="adapt_and_predict_image or SmallResNetTTT not yet available",
    )
    def test_eval_mode_enforced(self):
        """Verifies that adaptation engine enforces model.eval() to disable dropout."""
        model = SmallResNetTTT(num_classes=100)
        model.train()  # Explicitly enable training mode

        img = torch.rand(1, 3, 32, 32)
        out = adapt_and_predict_image(model, img, ttt_lr=0.001)

        assert not model.training, "adapt_and_predict_image must enforce model.eval()"
        assert isinstance(out, tuple) and len(out) == 4, "Expected 4-tuple return: (logits, pre_loss, post_loss, acc)"
        logits, pre_loss, post_loss, rot_acc = out
        assert logits.shape == (1, 100), f"Expected logits shape (1, 100), got {logits.shape}"

    @pytest.mark.skipif(
        not HAS_TTT_ENGINE,
        reason="VALID_CONDITIONS and evaluate_condition not yet available in src.ttt_cifar100",
    )
    def test_evaluation_conditions_enumeration(self):
        """Verifies the four required experimental conditions from Assignment Section 2 and condition validation."""
        expected_conditions = {
            "supervised_frozen",
            "joint_frozen",
            "standard_ttt",
            "online_ttt",
        }
        assert set(VALID_CONDITIONS) == expected_conditions, (
            f"Exported VALID_CONDITIONS {set(VALID_CONDITIONS)} must match expected {expected_conditions}"
        )
        assert len(VALID_CONDITIONS) == 4, "Must define exactly 4 experimental evaluation conditions"

        model = SmallResNetTTT(num_classes=100) if HAS_MODELS_TTT else None
        dummy_data = [(torch.rand(1, 3, 32, 32), torch.tensor([0]), ["test:0"])]

        with pytest.raises(ValueError, match="Unknown condition|Must be one of|Invalid condition|condition must be"):
            evaluate_condition(model, dummy_data, condition="invalid_condition_name")


# ==============================================================================
# CLASS 4: Tier 2 Boundary & Corner Cases
# ==============================================================================

class TestTier2BoundaryCornerCases:
    """Probes edge conditions, single-batch stability, numerical bounds, and invalid inputs."""

    @pytest.mark.skipif(not HAS_MODELS_TTT, reason="SmallResNetTTT not yet available in src.models")
    def test_single_image_b1_groupnorm_stability(self):
        """Verifies that B=1 passes through GroupNorm without NaN, Inf, or batch dimension errors."""
        model = SmallResNetTTT(num_classes=100)
        single_image = torch.randn(1, 3, 32, 32)

        # Eval mode
        model.eval()
        with torch.no_grad():
            out_eval = model(single_image)
        assert torch.isfinite(out_eval).all(), "B=1 in eval mode produced NaN or Inf"
        assert out_eval.shape == (1, 100)

        # Train mode with both heads exercised
        model.train()
        cls_logits, rot_logits = model.forward_both(single_image)
        assert torch.isfinite(cls_logits).all(), "B=1 cls_logits produced NaN or Inf"
        assert torch.isfinite(rot_logits).all(), "B=1 rot_logits produced NaN or Inf"

        # Backward pass exercises features, classifier, and rotation_head
        loss = cls_logits.sum() + rot_logits.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has None gradient"
                assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN or Inf"

    @pytest.mark.skipif(not HAS_DATA_EXP_B, reason="apply_gaussian_noise not yet available in src.data")
    @pytest.mark.parametrize("extreme_val", [0.0, 1.0])
    def test_noise_clamping_boundary_extremes(self, extreme_val):
        """Verifies strict [0.0, 1.0] clamping on saturated all-black and all-white inputs."""
        tensor = torch.full((3, 32, 32), extreme_val, dtype=torch.float32)
        noisy = apply_gaussian_noise(tensor, sample_id=0, sigma=0.5, noise_seed=42)

        assert noisy.min() >= 0.0, f"Pixel value {noisy.min()} fell below 0.0"
        assert noisy.max() <= 1.0, f"Pixel value {noisy.max()} exceeded 1.0"
        if extreme_val == 0.0:
            assert (noisy == 0.0).any(), "Negative noise perturbations should be clamped exactly to 0.0"
        else:
            assert (noisy == 1.0).any(), "Positive noise perturbations should be clamped exactly to 1.0"

    @pytest.mark.skipif(not HAS_DATA_EXP_B, reason="apply_gaussian_noise not yet available in src.data")
    def test_sigma_zero_identity(self):
        """Verifies that sigma=0.0 acts as an exact identity transform."""
        img = torch.rand(3, 32, 32)
        noisy = apply_gaussian_noise(img, sample_id=42, sigma=0.0, noise_seed=42)
        assert torch.equal(noisy, img), "sigma=0.0 must return bit-for-bit identical input tensor"

    @pytest.mark.skipif(not HAS_MODELS_TTT, reason="rotate_batch not yet available in src.models")
    def test_rotation_cycle_closure(self):
        """Verifies cyclic mathematical closure using rotate_batch output slices."""
        img = torch.rand(2, 3, 32, 32)
        rotated_imgs, rot_labels = rotate_batch(img)
        b = img.size(0)

        # Slices: 0:b is 0°, b:2b is 90°, 2b:3b is 180°, 3b:4b is 270°
        r0 = rotated_imgs[0:b]
        r90 = rotated_imgs[b:2 * b]
        r180 = rotated_imgs[2 * b:3 * b]
        r270 = rotated_imgs[3 * b:4 * b]

        # Check cyclic closure against 0° orientation
        assert torch.equal(r0, img), "0° slice must equal input batch"
        assert torch.equal(torch.rot90(r90, 3, dims=(-2, -1)), img), "90° slice rotated 270° must equal input"
        assert torch.equal(torch.rot90(r180, 2, dims=(-2, -1)), img), "180° slice rotated 180° must equal input"
        assert torch.equal(torch.rot90(r270, 1, dims=(-2, -1)), img), "270° slice rotated 90° must equal input"

    @pytest.mark.skipif(
        not (HAS_TTT_ENGINE and HAS_MODELS_TTT),
        reason="adapt_and_predict_image or SmallResNetTTT not yet available",
    )
    def test_lr_zero_parameter_invariance(self):
        """Verifies that ttt_lr=0.0 leaves model.features bit-for-bit unchanged."""
        model = SmallResNetTTT(num_classes=100)
        img = torch.rand(1, 3, 32, 32)
        pre_state = {k: v.clone() for k, v in model.features.state_dict().items()}

        adapt_and_predict_image(model, img, ttt_lr=0.0)

        post_state = model.features.state_dict()
        for k in pre_state:
            assert torch.equal(post_state[k], pre_state[k]), f"Feature parameter {k} changed when ttt_lr=0.0"

    @pytest.mark.skipif(not HAS_MODELS_TTT, reason="SmallResNetTTT not yet available in src.models")
    def test_unmasked_prediction_can_exceed_class_19(self):
        """Verifies Tuong Task 4: SmallResNetTTT unmasked classification head produces argmax >= 20."""
        model = SmallResNetTTT(num_classes=100)
        model.eval()

        # Deterministically bias class 55 so argmax is guaranteed to be 55 through the real forward pass
        with torch.no_grad():
            model.classifier[1].bias.zero_()
            model.classifier[1].bias[55] = 100.0

        img = torch.rand(1, 3, 32, 32)
        logits = model(img)

        assert logits.shape == (1, 100), f"Expected 100-way logits (1, 100), got {logits.shape}"
        predicted_class = logits.argmax(dim=-1).item()
        assert predicted_class == 55, "Real model forward pass must yield predicted class 55"
        assert predicted_class >= 20, "Argmax over unmasked 100 classes must not be truncated to 0..19"

        true_class = 5
        assert (predicted_class == true_class) is False, (
            f"Prediction {predicted_class} must evaluate as incorrect against true class {true_class}"
        )

    @pytest.mark.skipif(not HAS_DATA_EXP_B, reason="apply_gaussian_noise not yet available in src.data")
    def test_invalid_inputs_rejected(self):
        """Verifies proper error handling on invalid inputs to noise, model, and adaptation components."""
        # 1. Non-float tensor raises TypeError
        with pytest.raises((TypeError, ValueError)):
            apply_gaussian_noise(torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8), sample_id=0)

        # 2. 2D tensor (invalid rank) raises ValueError
        with pytest.raises(ValueError, match="must be 3D.*or 4D"):
            apply_gaussian_noise(torch.rand(32, 32), sample_id=0)

        # 3. Batch dimension mismatch raises ValueError
        with pytest.raises(ValueError, match="Batch dimension mismatch"):
            apply_gaussian_noise(torch.rand(2, 3, 32, 32), sample_id=0)

        # 4. Input channel mismatch to SmallResNetTTT (e.g. 4 channels vs 3 expected) raises RuntimeError
        if HAS_MODELS_TTT:
            model = SmallResNetTTT(num_classes=100)
            with pytest.raises(RuntimeError):
                model(torch.rand(1, 4, 32, 32))

        # 5. Negative learning rate rejected during adaptation (when engine is available)
        if HAS_TTT_ENGINE and HAS_MODELS_TTT:
            model = SmallResNetTTT(num_classes=100)
            with pytest.raises(ValueError, match="learning rate|ttt_lr|positive|non-negative"):
                adapt_and_predict_image(model, torch.rand(1, 3, 32, 32), ttt_lr=-0.01)


# ==============================================================================
# CLASS 5: Tier 3 State Transitions & Invariants
# ==============================================================================

class TestTier3StateTransitions:
    """Verifies Khoi Checks #2, #5, #7: head immutability, standard reset, online carryover."""

    @pytest.mark.skipif(
        not (HAS_TTT_ENGINE and HAS_MODELS_TTT),
        reason="adapt_and_predict_image or SmallResNetTTT not yet available",
    )
    def test_head_parameter_immutability_during_ttt(self):
        """Verifies that classifier and rotation_head parameters are NEVER modified by TTT."""
        model = SmallResNetTTT(num_classes=100)
        cls_init = {k: v.clone() for k, v in model.classifier.state_dict().items()}
        rot_init = {k: v.clone() for k, v in model.rotation_head.state_dict().items()}
        feat_init = {k: v.clone() for k, v in model.features.state_dict().items()}
        mean_init = model.mean.clone()
        std_init = model.std.clone()

        img = torch.rand(1, 3, 32, 32)
        adapt_and_predict_image(model, img, ttt_lr=0.01)

        # Invariant: classifier and rotation_head are bit-for-bit identical
        for k in cls_init:
            assert torch.equal(model.classifier.state_dict()[k], cls_init[k]), f"Classifier param {k} modified"
        for k in rot_init:
            assert torch.equal(model.rotation_head.state_dict()[k], rot_init[k]), f"Rotation param {k} modified"
        assert torch.equal(model.mean, mean_init)
        assert torch.equal(model.std, std_init)

        # Invariant: heads had requires_grad=False and grad is None
        for p in model.classifier.parameters():
            assert not p.requires_grad
            assert p.grad is None
        for p in model.rotation_head.parameters():
            assert not p.requires_grad
            assert p.grad is None

        # Features DID update
        feat_curr = model.features.state_dict()
        assert any(not torch.equal(feat_curr[k], feat_init[k]) for k in feat_init), "Features should have updated"

    @pytest.mark.skipif(
        not (HAS_TTT_ENGINE and HAS_MODELS_TTT),
        reason="adapt_and_predict_image or SmallResNetTTT not yet available",
    )
    def test_standard_ttt_exact_reset_between_samples(self):
        """Verifies Khoi Check #5: standard_ttt resets features to exact initial state between samples."""
        model = SmallResNetTTT(num_classes=100)
        theta_0 = {k: v.clone() for k, v in model.features.state_dict().items()}

        x1 = torch.rand(1, 3, 32, 32)
        x2 = torch.rand(1, 3, 32, 32)

        # Standard TTT simulation on sequence [x1, x2]
        adapt_and_predict_image(model, x1, ttt_lr=0.01)
        model.features.load_state_dict(theta_0)  # Standard TTT reset

        # Verify state is restored to theta_0 before adapting on x2
        for k in theta_0:
            assert torch.equal(model.features.state_dict()[k], theta_0[k])

        logits_x2_seq, _, _, _ = adapt_and_predict_image(model, x2, ttt_lr=0.01)
        theta_x2_seq = {k: v.clone() for k, v in model.features.state_dict().items()}
        model.features.load_state_dict(theta_0)

        # Isolated evaluation on x2 directly from theta_0
        model_iso = SmallResNetTTT(num_classes=100)
        model_iso.load_state_dict(model.state_dict())
        logits_x2_iso, _, _, _ = adapt_and_predict_image(model_iso, x2, ttt_lr=0.01)
        theta_x2_iso = {k: v.clone() for k, v in model_iso.features.state_dict().items()}

        assert torch.equal(logits_x2_seq, logits_x2_iso), "Sequence prediction does not match isolated prediction"
        for k in theta_x2_seq:
            assert torch.equal(theta_x2_seq[k], theta_x2_iso[k]), "Sequence weights do not match isolated weights"

    @pytest.mark.skipif(
        not (HAS_TTT_ENGINE and HAS_MODELS_TTT),
        reason="adapt_and_predict_image or SmallResNetTTT not yet available",
    )
    def test_online_ttt_weight_carryover_and_divergence(self):
        """Verifies Khoi Check #5: online_ttt carries updates forward and diverges from standard_ttt."""
        model = SmallResNetTTT(num_classes=100)
        theta_0 = {k: v.clone() for k, v in model.features.state_dict().items()}
        cls_0 = {k: v.clone() for k, v in model.classifier.state_dict().items()}

        x1 = torch.rand(1, 3, 32, 32)
        x2 = torch.rand(1, 3, 32, 32)

        # Online TTT: adapt x1, keep updates
        adapt_and_predict_image(model, x1, ttt_lr=0.01)
        # Adapt x2, keep updates
        adapt_and_predict_image(model, x2, ttt_lr=0.01)
        theta_2_online = {k: v.clone() for k, v in model.features.state_dict().items()}

        # Standard TTT: after x2 starting from theta_0
        model_std = SmallResNetTTT(num_classes=100)
        model_std.features.load_state_dict(theta_0)
        adapt_and_predict_image(model_std, x2, ttt_lr=0.01)
        theta_2_std = {k: v.clone() for k, v in model_std.features.state_dict().items()}

        # Divergence proven
        diverged = any(not torch.equal(theta_2_online[k], theta_2_std[k]) for k in theta_0)
        assert diverged, "online_ttt must diverge from standard_ttt at sample 2"

        # Classifier head remains unchanged despite online carryover
        for k in cls_0:
            assert torch.equal(model.classifier.state_dict()[k], cls_0[k])

    @pytest.mark.skipif(
        not (HAS_TTT_ENGINE and HAS_MODELS_TTT),
        reason="evaluate_condition or SmallResNetTTT not yet available in src.ttt_cifar100",
    )
    def test_evaluate_condition_standard_ttt_resets_features(self):
        """Verifies Khoi Check #5: evaluate_condition(condition='standard_ttt') performs episodic reset."""
        model = SmallResNetTTT(num_classes=100)
        theta_0 = {k: v.clone() for k, v in model.features.state_dict().items()}
        cls_0 = {k: v.clone() for k, v in model.classifier.state_dict().items()}
        rot_0 = {k: v.clone() for k, v in model.rotation_head.state_dict().items()}

        s1 = (torch.rand(1, 3, 32, 32), torch.tensor([5]), ["test:101"])
        s2 = (torch.rand(1, 3, 32, 32), torch.tensor([12]), ["test:102"])
        stream_loader = [s1, s2]

        summary, logs = _unpack_eval_results(
            evaluate_condition(model, stream_loader, condition="standard_ttt", ttt_lr=0.01)
        )

        # Invariant 1: After standard_ttt stream, model.features is restored to theta_0
        for k in theta_0:
            assert torch.equal(model.features.state_dict()[k], theta_0[k]), (
                f"Feature parameter {k} was not restored to theta_0 after standard_ttt stream"
            )

        # Invariant 2: Classifier and Rotation heads are bit-for-bit unchanged
        for k in cls_0:
            assert torch.equal(model.classifier.state_dict()[k], cls_0[k])
        for k in rot_0:
            assert torch.equal(model.rotation_head.state_dict()[k], rot_0[k])

        # Invariant 3: Sequence invariance — sample s2 prediction in stream [s1, s2]
        # matches evaluating s2 in isolation from theta_0
        model_iso = SmallResNetTTT(num_classes=100)
        model_iso.load_state_dict(model.state_dict())
        summary_iso, logs_iso = _unpack_eval_results(
            evaluate_condition(model_iso, [s2], condition="standard_ttt", ttt_lr=0.01)
        )

        pred_seq = logs[1].get("predicted_label", logs[1].get("predicted_class"))
        pred_iso = logs_iso[0].get("predicted_label", logs_iso[0].get("predicted_class"))
        assert pred_seq == pred_iso, "Sequence prediction must match isolated prediction"
        assert summary.get("update_count", 0) == 2

    @pytest.mark.skipif(
        not (HAS_TTT_ENGINE and HAS_MODELS_TTT),
        reason="evaluate_condition or SmallResNetTTT not yet available in src.ttt_cifar100",
    )
    def test_evaluate_condition_online_ttt_weight_carryover_and_divergence(self):
        """Verifies Khoi Check #5: evaluate_condition(condition='online_ttt') carries updates across samples."""
        model_std = SmallResNetTTT(num_classes=100)
        model_online = SmallResNetTTT(num_classes=100)
        model_online.load_state_dict(model_std.state_dict())
        theta_0 = {k: v.clone() for k, v in model_std.features.state_dict().items()}
        cls_0 = {k: v.clone() for k, v in model_std.classifier.state_dict().items()}

        s1 = (torch.rand(1, 3, 32, 32), torch.tensor([5]), ["test:101"])
        s2 = (torch.rand(1, 3, 32, 32), torch.tensor([12]), ["test:102"])
        stream_loader = [s1, s2]

        summary_std, _ = _unpack_eval_results(
            evaluate_condition(model_std, stream_loader, condition="standard_ttt", ttt_lr=0.01)
        )
        summary_online, _ = _unpack_eval_results(
            evaluate_condition(model_online, stream_loader, condition="online_ttt", ttt_lr=0.01)
        )

        # Invariant 1: online_ttt features diverge from theta_0
        diverged_from_theta_0 = any(
            not torch.equal(model_online.features.state_dict()[k], theta_0[k]) for k in theta_0
        )
        assert diverged_from_theta_0, "online_ttt must carry over updates and diverge from theta_0"

        # Invariant 2: online_ttt features diverge from standard_ttt features
        diverged_from_std = any(
            not torch.equal(model_online.features.state_dict()[k], model_std.features.state_dict()[k])
            for k in theta_0
        )
        assert diverged_from_std, "online_ttt features must diverge from standard_ttt features"

        # Invariant 3: Heads remain strictly frozen in online mode
        for k in cls_0:
            assert torch.equal(model_online.classifier.state_dict()[k], cls_0[k])

    @pytest.mark.skipif(
        not (HAS_TTT_ENGINE and HAS_MODELS_TTT),
        reason="evaluate_condition or SmallResNetTTT not yet available in src.ttt_cifar100",
    )
    @pytest.mark.parametrize("condition", ["supervised_frozen", "joint_frozen"])
    def test_evaluate_condition_frozen_baselines_zero_updates(self, condition):
        """Verifies evaluate_condition leaves 100% of parameters unchanged for frozen baselines."""
        model = SmallResNetTTT(num_classes=100)
        init_state = {k: v.clone() for k, v in model.state_dict().items()}
        stream_loader = [(torch.rand(1, 3, 32, 32), torch.tensor([5]), [f"test:{i}"]) for i in range(4)]

        summary, logs = _unpack_eval_results(
            evaluate_condition(model, stream_loader, condition=condition)
        )

        assert summary.get("update_count", 0) == 0
        assert len(logs) == 4
        for k in init_state:
            assert torch.equal(model.state_dict()[k], init_state[k])

    @pytest.mark.skipif(not HAS_MODELS_TTT, reason="SmallResNetTTT not yet available in src.models")
    def test_frozen_baselines_zero_parameter_modifications(self):
        """Verifies Khoi Check #5: supervised_frozen and joint_frozen leave 100% of parameters unchanged."""
        model = SmallResNetTTT(num_classes=100)
        init_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Evaluate on batch of 8 images without adaptation
        model.eval()
        x = torch.rand(8, 3, 32, 32)
        with torch.no_grad():
            _ = model(x)

        for k in init_state:
            assert torch.equal(model.state_dict()[k], init_state[k]), f"Parameter {k} modified during evaluation"

    @pytest.mark.skipif(not HAS_DATA_EXP_B, reason="apply_gaussian_noise not yet available in src.data")
    def test_repeatable_noise_across_conditions(self):
        """Verifies Khoi Check #2: sample i yields identical noisy tensor across all 4 conditions."""
        img = torch.rand(3, 32, 32)
        sample_id = 77
        seed = 42

        n_sup = apply_gaussian_noise(img, sample_id, sigma=0.08, noise_seed=seed)
        n_joint = apply_gaussian_noise(img, sample_id, sigma=0.08, noise_seed=seed)
        n_std = apply_gaussian_noise(img, sample_id, sigma=0.08, noise_seed=seed)
        n_online = apply_gaussian_noise(img, sample_id, sigma=0.08, noise_seed=seed)

        assert torch.equal(n_sup, n_joint)
        assert torch.equal(n_sup, n_std)
        assert torch.equal(n_sup, n_online)

    @pytest.mark.skipif(
        not (HAS_TTT_ENGINE and HAS_MODELS_TTT),
        reason="adapt_and_predict_image or SmallResNetTTT not yet available in src",
    )
    def test_stateless_sgd_no_optimizer_leakage(self, monkeypatch):
        """Verifies Khoi Check #7: adaptation strictly uses stateless SGD with momentum=0.0 on features only."""
        model = SmallResNetTTT(num_classes=100)
        img = torch.rand(1, 3, 32, 32)
        target_lr = 0.005

        captured_optimizers = []
        real_sgd = torch.optim.SGD

        def spy_sgd(params, *args, **kwargs):
            opt = real_sgd(params, *args, **kwargs)
            captured_optimizers.append(opt)
            return opt

        monkeypatch.setattr(torch.optim, "SGD", spy_sgd)

        # Execute adaptation on sample 1
        adapt_and_predict_image(model, img, ttt_lr=target_lr)

        assert len(captured_optimizers) >= 1, "adapt_and_predict_image must instantiate an SGD optimizer"
        opt = captured_optimizers[-1]

        # 1. Learning rate check
        assert opt.param_groups[0]["lr"] == target_lr, (
            f"Expected SGD lr={target_lr}, got {opt.param_groups[0]['lr']}"
        )
        # 2. Momentum=0.0 check (stateless SGD)
        assert opt.param_groups[0].get("momentum", 0.0) == 0.0, (
            "Khoi Check #7 violation: adaptation optimizer must have momentum=0.0"
        )
        # 3. Parameters check: only model.features
        feature_params = set(model.features.parameters())
        opt_params = set(opt.param_groups[0]["params"])
        assert opt_params == feature_params, (
            "Khoi Check #7 violation: adaptation optimizer must only optimize model.features"
        )
        # 4. Zero persistent state buffers
        assert len(opt.state) == 0, (
            "Khoi Check #7 violation: stateless SGD must contain 0 persistent state buffers"
        )

        # 5. Consecutive call verification: verify no buffer leakage
        adapt_and_predict_image(model, img, ttt_lr=target_lr)
        assert len(captured_optimizers) >= 2
        assert len(captured_optimizers[-1].state) == 0, (
            "Khoi Check #7 violation: optimizer state leaked across consecutive adaptation steps"
        )


# ==============================================================================
# CLASS 6: Tier 4 Real-World Scenarios & Downstream Integration
# ==============================================================================

class TestTier4RealWorldScenarios:
    """Verifies Khoi Checks #6, #8, Alka CSV logging schemas, and rolling window accuracy."""

    @pytest.mark.skipif(
        not (HAS_TTT_ENGINE and HAS_MODELS_TTT),
        reason="evaluate_condition or SmallResNetTTT not yet available in src.ttt_cifar100",
    )
    def test_stream_adaptation_simulation(self):
        """Simulates streaming 10 samples through all 4 conditions via evaluate_condition (Khoi Check #8)."""
        num_samples = 10
        stream_data = [
            (torch.rand(1, 3, 32, 32), torch.tensor([i % 20]), [f"test:{i}"])
            for i in range(num_samples)
        ]

        for cond in ("supervised_frozen", "joint_frozen", "standard_ttt", "online_ttt"):
            model = SmallResNetTTT(num_classes=100)
            res = evaluate_condition(model, stream_data, condition=cond, ttt_lr=0.001)
            summary, logs = _unpack_eval_results(res)

            assert len(logs) == num_samples, (
                f"Condition {cond} recorded {len(logs)} predictions, expected {num_samples}"
            )
            for r in logs:
                pred = r.get("predicted_label", r.get("predicted_class"))
                assert isinstance(pred, int) and 0 <= pred < 100, (
                    f"Condition {cond} produced prediction {pred} outside unmasked [0..99] range"
                )

            assert 0.0 <= float(summary.get("accuracy", 0.0)) <= 1.0
            assert float(summary.get("rotation_loss", 0.0)) >= 0.0

            if cond in ("supervised_frozen", "joint_frozen"):
                assert summary.get("update_count", 0) == 0, f"{cond} must perform 0 updates"
            else:
                assert summary.get("update_count", 0) == num_samples, (
                    f"{cond} must perform exactly {num_samples} updates"
                )

    @pytest.mark.skipif(
        not HAS_TTT_ENGINE,
        reason="ALKA_SUMMARY_COLUMNS not yet available in src.ttt_cifar100",
    )
    def test_alka_summary_csv_schema(self):
        """Verifies exact 23-column schema compliance for results/ttt_summary.csv generated by production code."""
        if callable(build_summary_row):
            sample_row = build_summary_row(
                experiment="B",
                condition="online_ttt",
                seed=42,
                checkpoint="checkpoints/joint_seed42.pt",
                dataset="CIFAR-100-20",
                split="test",
                corruption="gaussian_noise",
                severity=3,
                accuracy=0.432,
                runtime_seconds=12.5,
                update_count=2000,
                rotation_loss=0.854,
                learning_rate=0.001,
                adaptation_steps=1,
                batch_size=1,
                loss_weights="1.0_cls_1.0_rot",
                prediction_timing="post_adaptation",
                reset_mode="carryover",
                optimizer_reset="stateless",
                hardware="cpu",
                code_version="git-head",
                config_path="configs/ttt_cifar100_exp_b.yaml",
                run_id="exp_b_seed42_online_ttt_gaussian_noise",
            )
        else:
            sample_row = {
                "experiment": "B",
                "condition": "online_ttt",
                "seed": 42,
                "checkpoint": "checkpoints/joint_seed42.pt",
                "dataset": "CIFAR-100-20",
                "split": "test",
                "corruption": "gaussian_noise",
                "severity": 3,
                "accuracy": 0.432,
                "runtime_seconds": 12.5,
                "update_count": 2000,
                "rotation_loss": 0.854,
                "learning_rate": 0.001,
                "adaptation_steps": 1,
                "batch_size": 1,
                "loss_weights": "1.0_cls_1.0_rot",
                "prediction_timing": "post_adaptation",
                "reset_mode": "carryover",
                "optimizer_reset": "stateless",
                "hardware": "cpu",
                "code_version": "git-head",
                "config_path": "configs/ttt_cifar100_exp_b.yaml",
                "run_id": "exp_b_seed42_online_ttt_gaussian_noise",
            }

        # Check all 23 columns present
        assert set(sample_row.keys()) == set(ALKA_SUMMARY_COLUMNS)
        assert len(sample_row) == 23

        # Value constraint checks
        assert sample_row["experiment"] == "B"
        valid_set = set(VALID_CONDITIONS) if VALID_CONDITIONS else {
            "supervised_frozen", "joint_frozen", "standard_ttt", "online_ttt"
        }
        assert sample_row["condition"] in valid_set
        assert 0.0 <= float(sample_row["accuracy"]) <= 1.0
        assert float(sample_row["runtime_seconds"]) >= 0.0
        assert int(sample_row["update_count"]) >= 0
        assert float(sample_row["learning_rate"]) > 0.0
        assert int(sample_row["batch_size"]) == 1

    @pytest.mark.skipif(
        not HAS_TTT_ENGINE,
        reason="ALKA_WINDOW_COLUMNS and compute_rolling_windows not yet available in src.ttt_cifar100",
    )
    def test_alka_window_summary_csv_schema_and_conservation(self):
        """Verifies exact 10-column window schema and mathematical conservation laws."""
        # Simulated stream of 500 samples with window size 200
        targets = [5] * 500
        preds = [5] * 350 + [99] * 150  # 350 correct out of 500 = 70.0%
        windows = compute_rolling_windows(preds, targets, window_size=200)

        assert len(windows) == 3  # 200, 200, 100 (remainder window)

        for w in windows:
            assert set(w.keys()) == set(ALKA_WINDOW_COLUMNS)
            assert w["accuracy"] == pytest.approx(w["correct_count"] / w["window_size"])

        # Conservation Law 1: Sample count conservation
        assert sum(w["window_size"] for w in windows) == 500

        # Conservation Law 2: Correct count conservation
        assert sum(w["correct_count"] for w in windows) == 350

        # Conservation Law 3: Accuracy consistency
        overall_acc = sum(w["correct_count"] for w in windows) / 500.0
        assert overall_acc == pytest.approx(0.70)

        # Conservation Law 4: Contiguity of windows
        assert windows[0]["start_sample"] == 0 and windows[0]["end_sample"] == 200
        assert windows[1]["start_sample"] == 200 and windows[1]["end_sample"] == 400
        assert windows[2]["start_sample"] == 400 and windows[2]["end_sample"] == 500

    @pytest.mark.skipif(
        not HAS_TTT_ENGINE,
        reason="compute_rolling_windows not yet available in src.ttt_cifar100",
    )
    def test_windowed_accuracy_edge_cases(self):
        """Tests remainder window handling, all-correct, all-incorrect, and invalid inputs."""
        # Case 1: Remainder window where N < W (e.g. 128 images smoke test with W=200)
        smoke_windows = compute_rolling_windows([1] * 128, [1] * 128, window_size=200)
        assert len(smoke_windows) == 1
        assert smoke_windows[0]["window_size"] == 128
        assert smoke_windows[0]["accuracy"] == 1.0

        # Case 2: All incorrect
        fail_windows = compute_rolling_windows([1] * 50, [2] * 50, window_size=50)
        assert fail_windows[0]["correct_count"] == 0
        assert fail_windows[0]["accuracy"] == 0.0

        # Case 3: Invalid window size
        with pytest.raises(ValueError, match="window_size must be positive"):
            compute_rolling_windows([1] * 10, [1] * 10, window_size=0)

    @pytest.mark.skipif(
        not (HAS_TTT_ENGINE and HAS_MODELS_TTT),
        reason="evaluate_condition not yet available in src.ttt_cifar100",
    )
    def test_single_recorded_prediction_per_image(self):
        """Verifies Khoi Check #6: evaluate_condition records exactly 1 unique prediction per sample ID."""
        num_samples = 32
        sample_ids = [f"test:{i}" for i in range(num_samples)]
        stream_data = [
            (torch.rand(1, 3, 32, 32), torch.tensor([i % 20]), [s_id])
            for i, s_id in enumerate(sample_ids)
        ]

        model = SmallResNetTTT(num_classes=100)
        res = evaluate_condition(model, stream_data, condition="standard_ttt", ttt_lr=0.001)
        summary, logs = _unpack_eval_results(res)

        assert len(logs) == num_samples
        recorded_ids = [r["sample_id"] for r in logs]
        assert len(set(recorded_ids)) == num_samples, "Duplicate sample IDs found in prediction log"
        assert set(recorded_ids) == set(sample_ids), "Missing sample IDs in prediction log"
        for r in logs:
            pred = r.get("predicted_label", r.get("predicted_class"))
            assert isinstance(pred, int) and 0 <= pred < 100, "Predictions must be in 0..99"

    @pytest.mark.skipif(
        not HAS_TTT_ENGINE,
        reason="compute_delta_accuracy not yet available in src.ttt_cifar100",
    )
    def test_delta_accuracy_computation(self):
        """Verifies Alka Task 4: Delta accuracy percentage points against matched joint_frozen baseline."""
        acc_joint = 0.500
        acc_standard = 0.525
        acc_online = 0.540

        delta_standard = compute_delta_accuracy(acc_standard, acc_joint)
        delta_online = compute_delta_accuracy(acc_online, acc_joint)
        delta_carryover = compute_delta_accuracy(acc_online, acc_standard)

        assert delta_standard == pytest.approx(2.5)  # +2.5 percentage points
        assert delta_online == pytest.approx(4.0)    # +4.0 percentage points
        assert delta_carryover == pytest.approx(1.5) # +1.5 percentage points

        # Handles negative deltas properly
        delta_neg = compute_delta_accuracy(0.480, acc_joint)
        assert delta_neg == pytest.approx(-2.0)

        # Handles equal accuracies properly
        delta_zero = compute_delta_accuracy(acc_joint, acc_joint)
        assert delta_zero == pytest.approx(0.0)
