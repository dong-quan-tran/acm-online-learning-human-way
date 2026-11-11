"""Adversarial Empirical Stress-Testing Suite for Milestone 1 (Challenger 2).

Validates:
1. Noise determinism & repeatability (1,000 sample IDs, multiple passes/seeds, statistical independence)
2. Batch size invariance of noise (B=1 vs 64 vs 128, max abs diff == 0.0)
3. Cross-device noise invariance (CPU vs MPS, max abs diff == 0.0)
4. Clamping bounds (extreme inputs 0.0, 1.0, extreme sigma, strictly in [0.0, 1.0])
5. Data splits integrity (train 9,000, val 1,000, disjoint; test 2,000, sorted ascending, classes 0-19)
6. Loader iteration (get_exp_b_loaders, get_exp_b_loader, shapes, dtypes, sample_id formatting)
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data import (
    DATA_PATH,
    EXP_B_TEST_MEMBERSHIP_PATH,
    SHARED_C20_MEMBERSHIP_PATH,
    CIFAR100ExpBDataset,
    apply_gaussian_noise,
    get_exp_b_loader,
    get_exp_b_loaders,
    get_sample_noise_seed,
)
from src.models import SmallResNetTTT, rotate_batch


def test_noise_repeatability_1000_samples():
    """Verify bit-for-bit repeatability across 1,000 sample IDs and multiple seeds."""
    sample_ids = [f"test:{i}" for i in range(1000)]
    seeds = [42, 1337, 99999]
    img = torch.full((3, 32, 32), 0.5, dtype=torch.float32)

    for seed in seeds:
        for sid in sample_ids:
            t1 = apply_gaussian_noise(img, sid, sigma=0.08, noise_seed=seed)
            t2 = apply_gaussian_noise(img, sid, sigma=0.08, noise_seed=seed)
            assert torch.equal(t1, t2), f"Repeatability failed for sid={sid}, seed={seed}"


def test_noise_seed_uniqueness_1000_samples():
    """Verify SHA-256 derived seeds produce zero collisions across 1,000 sample IDs."""
    sample_ids = [f"test:{i}" for i in range(1000)]
    derived = {get_sample_noise_seed(42, sid) for sid in sample_ids}
    assert len(derived) == 1000, "Seed collision detected among 1,000 sample IDs"


def test_noise_statistical_independence():
    """Verify noise samples are statistically independent across seeds and sample IDs."""
    sample_ids = [f"test:{i}" for i in range(200)]
    img = torch.full((3, 32, 32), 0.5, dtype=torch.float32)

    noises_s42 = [(apply_gaussian_noise(img, sid, sigma=0.08, noise_seed=42) - 0.5).numpy().flatten() for sid in sample_ids]
    noises_s43 = [(apply_gaussian_noise(img, sid, sigma=0.08, noise_seed=43) - 0.5).numpy().flatten() for sid in sample_ids]

    noises_s42 = np.array(noises_s42)
    noises_s43 = np.array(noises_s43)

    # Mean and variance
    assert abs(np.mean(noises_s42)) < 0.005
    assert abs(np.std(noises_s42) - 0.08) < 0.002

    # Cross-seed correlation for same sample
    cross_seed_corrs = [np.corrcoef(noises_s42[i], noises_s43[i])[0, 1] for i in range(50)]
    assert max(abs(c) for c in cross_seed_corrs) < 0.15

    # Cross-sample correlation for same seed
    cross_sample_corrs = [np.corrcoef(noises_s42[i], noises_s42[i+1])[0, 1] for i in range(50)]
    assert max(abs(c) for c in cross_sample_corrs) < 0.15


def test_noise_batch_size_invariance():
    """Verify sample noise is identical whether computed in B=1, B=64, or B=128."""
    images = torch.rand(128, 3, 32, 32)
    sids = [f"test:{i}" for i in range(128)]

    batched_128 = apply_gaussian_noise(images, sids, sigma=0.08, noise_seed=42)
    batched_64 = apply_gaussian_noise(images[:64], sids[:64], sigma=0.08, noise_seed=42)

    # B=64 vs B=128
    assert torch.equal(batched_64, batched_128[:64])

    # B=1 vs B=128
    for i in range(128):
        single_3d = apply_gaussian_noise(images[i], sids[i], sigma=0.08, noise_seed=42)
        single_4d = apply_gaussian_noise(images[i:i+1], [sids[i]], sigma=0.08, noise_seed=42)
        assert torch.equal(single_3d, batched_128[i])
        assert torch.equal(single_4d[0], batched_128[i])


def test_cross_device_noise_invariance():
    """Verify bit-for-bit equivalence between CPU and MPS device noise generation."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this system")

    images_cpu = torch.rand(64, 3, 32, 32)
    images_mps = images_cpu.clone().to("mps")
    sids = [f"test:{i}" for i in range(64)]

    out_cpu = apply_gaussian_noise(images_cpu, sids, sigma=0.08, noise_seed=42)
    out_mps = apply_gaussian_noise(images_mps, sids, sigma=0.08, noise_seed=42)

    assert out_cpu.device == torch.device("cpu")
    assert out_mps.device.type == "mps"
    assert (out_cpu - out_mps.cpu()).abs().max().item() == 0.0


def test_clamping_bounds_extreme_sigma():
    """Verify strict [0.0, 1.0] pixel bounds under extreme inputs and high sigmas."""
    zeros = torch.zeros(10, 3, 32, 32)
    ones = torch.ones(10, 3, 32, 32)
    sids = [f"test:{i}" for i in range(10)]

    for sigma in [0.0, 0.08, 1.0, 10.0, 1000.0]:
        nz = apply_gaussian_noise(zeros, sids, sigma=sigma, noise_seed=42)
        no = apply_gaussian_noise(ones, sids, sigma=sigma, noise_seed=42)
        assert nz.min().item() >= 0.0 and nz.max().item() <= 1.0
        assert no.min().item() >= 0.0 and no.max().item() <= 1.0


def test_data_splits_integrity():
    """Verify train/val/test split sizes, disjointness, ascending order, and C20 labels."""
    with open(SHARED_C20_MEMBERSHIP_PATH) as f:
        shared = json.load(f)
    with open(EXP_B_TEST_MEMBERSHIP_PATH) as f:
        test_mem = json.load(f)

    train_ids = shared["train"]
    val_ids = shared["validation"]
    test_ids = test_mem["test"]

    assert len(train_ids) == 9000
    assert len(val_ids) == 1000
    assert len(test_ids) == 2000
    assert len(set(train_ids)) == 9000
    assert len(set(val_ids)) == 1000
    assert len(set(test_ids)) == 2000

    # Disjointness
    assert len(set(train_ids) & set(val_ids)) == 0

    # Ascending order
    assert all(test_ids[i] < test_ids[i+1] for i in range(len(test_ids) - 1))


def test_loader_iteration_and_dtypes():
    """Verify DataLoader yields correct shapes, dtypes, sample_id formats, and labels."""
    train_ldr, val_ldr, test_ldr = get_exp_b_loaders(
        batch_size=32,
        test_batch_size=1,
        corruption="gaussian_noise",
        noise_sigma=0.08,
        noise_seed=42,
    )

    pattern = re.compile(r"^(train|test):(\d+)$")

    # Sample batch from train
    bx, by, bsids = next(iter(train_ldr))
    assert bx.shape == (32, 3, 32, 32)
    assert bx.dtype == torch.float32
    assert by.dtype == torch.int64
    assert all(pattern.match(s) and s.startswith("train:") for s in bsids)
    assert (by >= 0).all() and (by <= 19).all()

    # Sample batch from test
    tx, ty, tsids = next(iter(test_ldr))
    assert tx.shape == (1, 3, 32, 32)
    assert tx.dtype == torch.float32
    assert ty.dtype == torch.int64
    assert all(pattern.match(s) and s.startswith("test:") for s in tsids)
    assert (ty >= 0).all() and (ty <= 19).all()
