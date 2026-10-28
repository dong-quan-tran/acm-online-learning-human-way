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
