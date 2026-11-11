"""SmallResNet architecture extracted from the existing CIFAR notebook.

MNIST defaults: one input channel, ten outputs, ToTensor input in [0, 1].
Normalization is explicit and happens only here (identity by default).
"""
import torch
from torch import nn


def conv_bn(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                         nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(conv_bn(channels, channels),
                                   conv_bn(channels, channels))

    def forward(self, x):
        return x + self.block(x)


class SmallResNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, dropout=0.2,
                 mean=None, std=None):
        super().__init__()
        mean = [0.0] * in_channels if mean is None else mean
        std = [1.0] * in_channels if std is None else std
        if len(mean) != in_channels or len(std) != in_channels or any(s <= 0 for s in std):
            raise ValueError("mean/std must match input channels and std must be positive")
        self.register_buffer("mean", torch.tensor(mean).view(1, in_channels, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, in_channels, 1, 1))
        self.features = nn.Sequential(
            conv_bn(in_channels, 64), Residual(64), nn.MaxPool2d(2),
            conv_bn(64, 128), Residual(128), nn.MaxPool2d(2),
            conv_bn(128, 256), Residual(256), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.features((x - self.mean) / self.std).mean(dim=(2, 3))
        return self.classifier(x)


# ==============================================================================
# Experiment B (TTT): GroupNorm backbone with 100-way cls & 4-way rotation heads
# ==============================================================================

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def conv_gn(c_in, c_out, num_groups=16):
    """3x3 convolution followed by GroupNorm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.GroupNorm(num_groups, c_out),
        nn.ReLU(inplace=True),
    )


class ResidualGN(nn.Module):
    """Two-layer conv_gn residual block: x + block(x)."""

    def __init__(self, channels, num_groups=16):
        super().__init__()
        self.block = nn.Sequential(
            conv_gn(channels, channels, num_groups=num_groups),
            conv_gn(channels, channels, num_groups=num_groups),
        )

    def forward(self, x):
        return x + self.block(x)


def rotate_batch(images):
    """Expand (B, C, H, W) into (4*B, C, H, W) for [0°, 90°, 180°, 270°] rotations.

    Returns:
        rotated: Tensor of shape (4*B, C, H, W) with rotations concatenated along dim 0.
        labels: LongTensor of shape (4*B,) with values [0, 1, 2, 3] repeated per orientation.
    """
    batch_size = images.size(0)
    rotated = torch.cat([torch.rot90(images, k, dims=(-2, -1)) for k in range(4)], dim=0)
    labels = torch.arange(4, device=images.device, dtype=torch.long).repeat_interleave(batch_size)
    return rotated, labels


class SmallResNetTTT(nn.Module):
    """Shared GroupNorm SmallResNet backbone for Test-Time Training (TTT).

    Features:
    - GroupNorm(16, C) across all stages, with 0 BatchNorm2d layers.
    - Unmasked 100-way linear classifier (evaluates on classes 0..19 without masking).
    - 4-way linear rotation prediction head (0°, 90°, 180°, 270°).
    - Internal per-channel mean and std normalization via registered buffers.
    """

    def __init__(
        self,
        num_classes=100,
        in_channels=3,
        dropout=0.2,
        num_groups=16,
        mean=CIFAR100_MEAN,
        std=CIFAR100_STD,
    ):
        super().__init__()
        mean = CIFAR100_MEAN if mean is None else mean
        std = CIFAR100_STD if std is None else std
        if len(mean) != in_channels or len(std) != in_channels or any(s <= 0 for s in std):
            raise ValueError("mean/std must match input channels and std must be positive")
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).view(1, in_channels, 1, 1))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).view(1, in_channels, 1, 1))
        self.features = nn.Sequential(
            conv_gn(in_channels, 64, num_groups), ResidualGN(64, num_groups), nn.MaxPool2d(2),
            conv_gn(64, 128, num_groups), ResidualGN(128, num_groups), nn.MaxPool2d(2),
            conv_gn(128, 256, num_groups), ResidualGN(256, num_groups), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, num_classes))
        self.rotation_head = nn.Linear(256, 4)

    def extract_features(self, x):
        """Normalizes input and pools through backbone to produce (B, 256) representation."""
        normalized = (x - self.mean) / self.std
        return self.features(normalized).mean(dim=(2, 3))

    def forward(self, x):
        """Standard classification forward pass returning (B, num_classes) logits."""
        return self.classifier(self.extract_features(x))

    def forward_rotation(self, x):
        """Rotation forward pass returning (B, 4) logits."""
        return self.rotation_head(self.extract_features(x))

    def forward_both(self, x):
        """Joint forward pass returning (class_logits, rotation_logits)."""
        feats = self.extract_features(x)
        return self.classifier(feats), self.rotation_head(feats)

