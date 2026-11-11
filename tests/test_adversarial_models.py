"""Adversarial stress-testing suite for src/models.py (Milestone 1).

Executed by Challenger 1 (challenger_m1_1) to empirically verify:
1. Complete absence of BatchNorm across all layers/submodules and presence of GroupNorm(16, C).
2. Strict batch-size invariance between B=1 and B>1 (eval mode and feature train mode).
3. Rotation batching stress-test across varying batch sizes, orientation cycling, and spatial pixel movement.
4. Head isolation and gradient orthogonality between classification and rotation tasks.
5. Inplace ReLU autograd safety during backprop through residual blocks.
6. Buffer registration and input validation.
"""

import pytest
import torch
import torch.nn as nn
from src.models import (
    CIFAR100_MEAN,
    CIFAR100_STD,
    ResidualGN,
    SmallResNet,
    SmallResNetTTT,
    conv_gn,
    rotate_batch,
)


class TestAdversarialGroupNormVsBatchNorm:
    """Stress-test architectural norm requirements."""

    def test_zero_batchnorm_exhaustive(self):
        """Verify 0 instances of any BatchNorm variant across entire model hierarchy."""
        model = SmallResNetTTT()

        # Check all possible BatchNorm variants in PyTorch
        prohibited_norm_classes = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.modules.batchnorm._BatchNorm,
            nn.modules.batchnorm._NormBase,
        )

        found_batchnorms = []
        for name, module in model.named_modules():
            if isinstance(module, prohibited_norm_classes):
                found_batchnorms.append((name, type(module).__name__))

        assert len(found_batchnorms) == 0, f"Found prohibited BatchNorm modules: {found_batchnorms}"

    def test_groupnorm_layer_count_and_config(self):
        """Verify exact count of GroupNorm layers and num_groups == 16 for all of them."""
        model = SmallResNetTTT()

        groupnorm_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.GroupNorm):
                groupnorm_layers.append((name, module))

        # 3 stages * (1 conv_gn + 2 conv_gn inside ResidualGN) = 9 GroupNorm layers
        assert len(groupnorm_layers) == 9, f"Expected exactly 9 GroupNorm layers, found {len(groupnorm_layers)}"

        expected_channels = {
            "features.0.1": 64,
            "features.1.block.0.1": 64,
            "features.1.block.1.1": 64,
            "features.3.1": 128,
            "features.4.block.0.1": 128,
            "features.4.block.1.1": 128,
            "features.6.1": 256,
            "features.7.block.0.1": 256,
            "features.7.block.1.1": 256,
        }

        for name, gn in groupnorm_layers:
            assert gn.num_groups == 16, f"Layer {name} has num_groups={gn.num_groups}, expected 16"
            assert gn.num_channels % 16 == 0, f"Layer {name} num_channels={gn.num_channels} not divisible by 16"
            if name in expected_channels:
                assert gn.num_channels == expected_channels[name], (
                    f"Layer {name} expected {expected_channels[name]} channels, got {gn.num_channels}"
                )
            assert gn.affine is True, f"Layer {name} must have learnable affine parameters"
            assert gn.weight is not None and gn.bias is not None

    def test_groupnorm_layer_exact_zero_batch_dependence(self):
        """Verify that GroupNorm itself has mathematically zero batch dependence."""
        gn = nn.GroupNorm(16, 64)
        x = torch.rand(64, 64, 16, 16)
        out_batch = gn(x)
        out_single = gn(x[0:1])
        diff = (out_batch[0:1] - out_single).abs().max().item()
        assert diff == 0.0, f"GroupNorm alone must have exact 0.0 batch difference, got {diff}"

    def test_existing_smallresnet_untouched(self):
        """Ensure original SmallResNet still exists and uses BatchNorm (no accidental regression)."""
        original_model = SmallResNet(num_classes=10, in_channels=1)
        bn_layers = [m for m in original_model.modules() if isinstance(m, nn.BatchNorm2d)]
        assert len(bn_layers) == 9, "Original SmallResNet must preserve its 9 BatchNorm2d layers"


class TestAdversarialBatchSizeInvariance:
    """Stress-test numerical batch-size invariance of SmallResNetTTT."""

    @pytest.mark.parametrize("batch_size", [2, 3, 7, 13, 32, 64])
    def test_eval_mode_batch_invariance_against_single_sample_float32(self, batch_size):
        """Compare output of sample i evaluated alone vs inside batch of size B.

        GroupNorm normalizes each sample independently across channel groups.
        Across 9 convolutions, GEMM FMA parallel reduction order in CPU BLAS introduces
        minor floating-point rounding variance (~2e-6), which is strictly within 1e-5.
        """
        torch.manual_seed(42 + batch_size)
        model = SmallResNetTTT()
        model.eval()

        batch_x = torch.rand(batch_size, 3, 32, 32)

        with torch.no_grad():
            batch_cls = model(batch_x)
            batch_rot = model.forward_rotation(batch_x)
            batch_feat = model.extract_features(batch_x)

            for i in range(batch_size):
                single_x = batch_x[i : i + 1]
                single_cls = model(single_x)
                single_rot = model.forward_rotation(single_x)
                single_feat = model.extract_features(single_x)

                diff_cls = (batch_cls[i : i + 1] - single_cls).abs().max().item()
                diff_rot = (batch_rot[i : i + 1] - single_rot).abs().max().item()
                diff_feat = (batch_feat[i : i + 1] - single_feat).abs().max().item()

                assert diff_cls <= 1e-5, f"Classification diff for sample {i}/{batch_size} exceeded 1e-5: {diff_cls}"
                assert diff_rot <= 1e-5, f"Rotation diff for sample {i}/{batch_size} exceeded 1e-5: {diff_rot}"
                assert diff_feat <= 1e-5, f"Feature diff for sample {i}/{batch_size} exceeded 1e-5: {diff_feat}"

    def test_double_precision_batch_invariance_proves_zero_leakage(self):
        """Under float64, GEMM rounding drops to machine epsilon (~1e-15), proving zero structural coupling."""
        torch.manual_seed(1234)
        model = SmallResNetTTT().double().eval()

        batch_size = 32
        batch_x = torch.rand(batch_size, 3, 32, 32, dtype=torch.float64)

        with torch.no_grad():
            batch_cls = model(batch_x)
            for i in [0, 1, 15, 31]:
                single_x = batch_x[i : i + 1]
                single_cls = model(single_x)
                diff_cls = (batch_cls[i : i + 1] - single_cls).abs().max().item()
                assert diff_cls < 1e-12, f"Float64 batch diff {diff_cls} exceeded 1e-12"

    def test_batchnorm_contrast_demonstrating_collapse(self):
        """Demonstrate that BatchNorm has massive (> 1.0) batch coupling in train mode, whereas GroupNorm does not."""
        torch.manual_seed(42)
        bn_model = SmallResNet(num_classes=100, in_channels=3).train()
        gn_model = SmallResNetTTT().train()

        x = torch.rand(32, 3, 32, 32)
        diff_bn = (bn_model(x)[0:1] - bn_model(x[0:1])).abs().max().item()
        diff_gn_feat = (gn_model.extract_features(x)[0:1] - gn_model.extract_features(x[0:1])).abs().max().item()

        assert diff_bn > 1.0, f"BatchNorm expected massive batch coupling (>1.0), got {diff_bn}"
        assert diff_gn_feat < 1e-5, f"GroupNorm features must have negligible diff (<1e-5), got {diff_gn_feat}"


class TestAdversarialRotationBatching:
    """Stress-test rotate_batch function."""

    @pytest.mark.parametrize("b", [1, 2, 7, 16])
    def test_rotate_batch_shapes_and_labels(self, b):
        """Verify shape (4B, C, H, W) and label structure [0]*B + [1]*B + [2]*B + [3]*B."""
        torch.manual_seed(99 + b)
        images = torch.rand(b, 3, 32, 32)
        rot_x, rot_y = rotate_batch(images)

        assert rot_x.shape == (4 * b, 3, 32, 32)
        assert rot_y.shape == (4 * b,)
        assert rot_y.dtype == torch.long
        assert rot_y.device == images.device

        expected_labels = torch.tensor([0] * b + [1] * b + [2] * b + [3] * b, dtype=torch.long, device=images.device)
        assert torch.equal(rot_y, expected_labels), f"Label sequence mismatch for B={b}"

    def test_rotation_mathematical_identities(self):
        """Verify 0° is exact identity, 360° is identity, and k*90° matches torch.rot90."""
        b = 5
        images = torch.rand(b, 3, 32, 32)
        rot_x, _ = rotate_batch(images)

        # 0° rotation must be bit-exact identity
        assert torch.equal(rot_x[:b], images), "0° rotation must be identical to input"

        # 90° rotation matches rot90(images, 1, (-2, -1))
        assert torch.equal(rot_x[b : 2 * b], torch.rot90(images, 1, dims=(-2, -1)))

        # 180° rotation matches rot90(images, 2, (-2, -1))
        assert torch.equal(rot_x[2 * b : 3 * b], torch.rot90(images, 2, dims=(-2, -1)))

        # 270° rotation matches rot90(images, 3, (-2, -1))
        assert torch.equal(rot_x[3 * b :], torch.rot90(images, 3, dims=(-2, -1)))

        # Four consecutive 90° rotations return to original
        r90_4 = torch.rot90(rot_x[3 * b :], 1, dims=(-2, -1))
        assert torch.equal(r90_4, images), "Four 90° rotations must return to original image"

    def test_spatial_pixel_displacement(self):
        """Verify rotation alters spatial coordinates (H, W) and does NOT permute channels."""
        # Create an image with known asymmetric landmark: single bright pixel at top-left (0, 0) in channel 0
        img = torch.zeros(1, 3, 8, 8)
        img[0, 0, 0, 0] = 10.0  # Top-left of channel 0

        rot_x, _ = rotate_batch(img)

        # For torch.rot90(..., k=1, dims=(-2, -1)), rotation is from -1 axis to -2 axis (counter-clockwise)
        # (0, 0) -> (7, 0) for k=1
        # (0, 0) -> (7, 7) for k=2
        # (0, 0) -> (0, 7) for k=3
        assert rot_x[0, 0, 0, 0].item() == 10.0, "k=0 landmark wrong"
        assert rot_x[1, 0, 7, 0].item() == 10.0, "k=1 landmark wrong"
        assert rot_x[2, 0, 7, 7].item() == 10.0, "k=2 landmark wrong"
        assert rot_x[3, 0, 0, 7].item() == 10.0, "k=3 landmark wrong"

        # Channels 1 and 2 must remain exactly zero everywhere across all 4 rotations
        assert rot_x[:, 1:, :, :].abs().max().item() == 0.0, "Rotation contaminated other channels"


class TestAdversarialHeadIsolationAndAutograd:
    """Stress-test head isolation, unmasked logits, and autograd gradient orthogonality."""

    def test_unmasked_100_logits_and_4_rotation_logits(self):
        """Verify model output dimensions and unmasked predictions."""
        model = SmallResNetTTT()
        model.eval()

        x = torch.rand(4, 3, 32, 32)
        cls_logits = model(x)
        rot_logits = model.forward_rotation(x)
        c, r = model.forward_both(x)

        assert cls_logits.shape == (4, 100), f"Expected (4, 100), got {cls_logits.shape}"
        assert rot_logits.shape == (4, 4), f"Expected (4, 4), got {rot_logits.shape}"
        assert torch.equal(cls_logits, c), "forward_both class logits differ from forward"
        assert torch.equal(rot_logits, r), "forward_both rot logits differ from forward_rotation"

        # Check unmasked: predictions across all 100 classes are active (variance > 0 across classes 20-99)
        assert cls_logits[:, 20:].var().item() > 0.0, "Logits for classes 20-99 should be unmasked"

    def test_gradient_orthogonality_between_heads(self):
        """Verify backprop through rotation head does NOT create gradients in classifier head, and vice-versa."""
        model = SmallResNetTTT()
        x = torch.rand(2, 3, 32, 32)

        # 1. Backprop rotation loss
        model.zero_grad(set_to_none=True)
        rot_loss = model.forward_rotation(x).sum()
        rot_loss.backward()

        # Classification head must have NO gradient
        for p in model.classifier.parameters():
            assert p.grad is None, "Rotation loss produced unexpected gradient in classification head"

        # Rotation head and backbone MUST have gradients
        for p in model.rotation_head.parameters():
            assert p.grad is not None, "Rotation head missing gradient from rotation loss"

        first_conv_weight = model.features[0][0].weight
        assert first_conv_weight.grad is not None, "Backbone features missing gradient from rotation loss"

        # 2. Backprop classification loss
        model.zero_grad(set_to_none=True)
        cls_loss = model(x).sum()
        cls_loss.backward()

        # Rotation head must have NO gradient
        for p in model.rotation_head.parameters():
            assert p.grad is None, "Classification loss produced unexpected gradient in rotation head"

        # Classification head MUST have gradients
        for p in model.classifier.parameters():
            assert p.grad is not None, "Classification head missing gradient from classification loss"

    def test_inplace_relu_residual_backprop_safety(self):
        """Stress-test inplace=True in conv_gn and ResidualGN during TTT adaptation steps.

        In some architectures, inplace ReLU causes autograd RuntimeError when backpropagating
        through residual connections. Verify that full SGD step works cleanly without autograd error.
        """
        model = SmallResNetTTT()

        # Simulate TTT adaptation: freeze heads, optimize features only
        for p in model.classifier.parameters():
            p.requires_grad = False
        for p in model.rotation_head.parameters():
            p.requires_grad = False
        for p in model.features.parameters():
            p.requires_grad = True

        optimizer = torch.optim.SGD(model.features.parameters(), lr=0.01, momentum=0.0)

        # Snapshot initial weights
        initial_feat_weight = model.features[0][0].weight.clone()
        initial_cls_weight = model.classifier[1].weight.clone()
        initial_rot_weight = model.rotation_head.weight.clone()

        # Run 3 consecutive adaptation steps on rotated views
        x = torch.rand(2, 3, 32, 32)
        criterion = nn.CrossEntropyLoss()

        for step in range(3):
            optimizer.zero_grad()
            rot_x, rot_y = rotate_batch(x)
            rot_logits = model.forward_rotation(rot_x)
            loss = criterion(rot_logits, rot_y)
            loss.backward()
            optimizer.step()

        # Verify feature weights updated
        assert not torch.equal(model.features[0][0].weight, initial_feat_weight), "Features did not update"

        # Verify frozen head weights remain bit-exact identical
        assert torch.equal(model.classifier[1].weight, initial_cls_weight), "Classifier was modified during TTT!"
        assert torch.equal(model.rotation_head.weight, initial_rot_weight), "Rotation head was modified during TTT!"


class TestAdversarialBuffersAndValidation:
    """Stress-test buffer registration, device moves, and parameter counts."""

    def test_buffers_not_in_parameters(self):
        """Mean and std must be registered buffers, NOT trainable parameters."""
        model = SmallResNetTTT()
        param_names = [name for name, _ in model.named_parameters()]
        buffer_names = [name for name, _ in model.named_buffers()]

        assert "mean" not in param_names, "mean should not be a parameter"
        assert "std" not in param_names, "std should not be a parameter"
        assert "mean" in buffer_names, "mean must be a registered buffer"
        assert "std" in buffer_names, "std must be a registered buffer"

        # Check values match CIFAR100 constants
        assert torch.allclose(model.mean.squeeze(), torch.tensor(CIFAR100_MEAN))
        assert torch.allclose(model.std.squeeze(), torch.tensor(CIFAR100_STD))

    def test_invalid_constructor_inputs(self):
        """Verify model constructor rejects invalid channels, non-positive std, etc."""
        with pytest.raises(ValueError):
            # std <= 0
            SmallResNetTTT(std=(0.2, -0.1, 0.2))

        with pytest.raises(ValueError):
            # mean channel length mismatch
            SmallResNetTTT(mean=(0.5, 0.5))
