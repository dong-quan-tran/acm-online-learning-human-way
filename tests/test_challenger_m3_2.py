"""Challenger 2 Empirical Verification Suite for Milestone 3 (Experiment B).

Checks:
1. Checkpoint Architecture Inspection:
   - exp_b_supervised_seed42.pt and exp_b_joint_seed42.pt
   - Both instantiate SmallResNetTTT cleanly
   - 0 BatchNorm2d layers across all submodules
   - Exactly 9 GroupNorm layers, all with num_groups=16
   - classifier is Linear(256, 100) (unmasked)
   - rotation_head is Linear(256, 4)
   - metadata verification (epoch, mode, seed, val_acc, val_rot_acc, source_train_seconds)

2. Label Isolation & Signature Challenge:
   - inspect.signature on adapt_and_predict_image strictly accepts (model, image, ttt_lr, ttt_steps, lambda_rot)
   - rejects any parameter named label, target, y, ground_truth, etc.
   - runtime verification: classifier.weight and rotation_head.weight have zero change
     (torch.max(torch.abs(w_before - w_after)) == 0.0)

3. Reset vs Carryover Verification:
   - standard_ttt on sample 1 and sample 2 resets feature backbone to identical joint checkpoint state
   - online_ttt on sample 1 and sample 2 accumulates weight updates in model.features
"""

import copy
import inspect
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.models import SmallResNetTTT
from src.ttt_cifar100 import (
    adapt_and_predict_image,
    evaluate_condition,
    load_model,
)


CHECKPOINT_DIR = Path("checkpoints")
SUPERVISED_CKPT = CHECKPOINT_DIR / "exp_b_supervised_seed42.pt"
JOINT_CKPT = CHECKPOINT_DIR / "exp_b_joint_seed42.pt"


def test_checkpoint_files_exist():
    assert SUPERVISED_CKPT.exists(), f"Missing {SUPERVISED_CKPT}"
    assert JOINT_CKPT.exists(), f"Missing {JOINT_CKPT}"


@pytest.mark.parametrize("ckpt_path,expected_mode", [
    (SUPERVISED_CKPT, "supervised"),
    (JOINT_CKPT, "joint"),
])
def test_checkpoint_architecture_inspection(ckpt_path, expected_mode):
    model, metadata = load_model(ckpt_path, device="cpu", num_classes=100)
    assert isinstance(model, SmallResNetTTT), f"Model is not SmallResNetTTT: {type(model)}"

    # 1. Verify 0 BatchNorm2d
    bn_layers = [name for name, mod in model.named_modules() if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    assert len(bn_layers) == 0, f"Found BatchNorm layers in {ckpt_path}: {bn_layers}"

    # 2. Verify exactly 9 GroupNorm layers, all with num_groups=16
    gn_layers = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, nn.GroupNorm)]
    assert len(gn_layers) == 9, f"Expected 9 GroupNorm layers, found {len(gn_layers)}: {[n for n, _ in gn_layers]}"
    for name, gn in gn_layers:
        assert gn.num_groups == 16, f"Layer {name} num_groups={gn.num_groups}, expected 16"

    # 3. Verify classifier is nn.Sequential(nn.Dropout, nn.Linear(256, 100))
    assert hasattr(model, "classifier"), "Model has no classifier attribute"
    assert isinstance(model.classifier, nn.Sequential), "classifier is not nn.Sequential"
    linear_cls = [m for m in model.classifier.modules() if isinstance(m, nn.Linear)]
    assert len(linear_cls) == 1, f"Expected 1 Linear layer in classifier, found {len(linear_cls)}"
    assert linear_cls[0].in_features == 256, f"classifier in_features={linear_cls[0].in_features}, expected 256"
    assert linear_cls[0].out_features == 100, f"classifier out_features={linear_cls[0].out_features}, expected 100"

    # 4. Verify rotation_head is Linear(256, 4)
    assert hasattr(model, "rotation_head"), "Model has no rotation_head attribute"
    assert isinstance(model.rotation_head, nn.Linear), f"rotation_head is not nn.Linear: {type(model.rotation_head)}"
    assert model.rotation_head.in_features == 256, f"rotation_head in_features={model.rotation_head.in_features}, expected 256"
    assert model.rotation_head.out_features == 4, f"rotation_head out_features={model.rotation_head.out_features}, expected 4"

    # 5. Verify checkpoint metadata
    required_keys = ["epoch", "mode", "seed", "val_acc", "val_rot_acc", "source_train_seconds"]
    for key in required_keys:
        assert key in metadata, f"Metadata missing key '{key}' in {ckpt_path}. Found keys: {list(metadata.keys())}"

    assert metadata["mode"] == expected_mode, f"Expected mode {expected_mode}, got {metadata['mode']}"
    assert metadata["seed"] == 42, f"Expected seed 42, got {metadata['seed']}"
    assert metadata["epoch"] == 15, f"Expected 15 epochs, got {metadata['epoch']}"
    assert isinstance(metadata["val_acc"], (float, int)), f"Invalid val_acc: {metadata['val_acc']}"
    assert isinstance(metadata["val_rot_acc"], (float, int)), f"Invalid val_rot_acc: {metadata['val_rot_acc']}"
    assert isinstance(metadata["source_train_seconds"], (float, int)), f"Invalid source_train_seconds: {metadata['source_train_seconds']}"
    assert metadata["source_train_seconds"] > 0, "source_train_seconds must be positive"


def test_label_isolation_signature():
    sig = inspect.signature(adapt_and_predict_image)
    param_names = list(sig.parameters.keys())
    expected_params = ["model", "image", "ttt_lr", "ttt_steps", "lambda_rot"]
    assert param_names == expected_params, f"adapt_and_predict_image signature mismatch! Expected {expected_params}, got {param_names}"

    # Verify no varargs (*args) or varkw (**kwargs) which could smuggle labels
    for p in sig.parameters.values():
        assert p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD), (
            f"Signature accepts varargs/varkw parameter {p.name}, violating strict label isolation!"
        )

    # Explicit check for forbidden parameter names
    forbidden = ["label", "labels", "target", "targets", "y", "gt", "ground_truth", "class_idx"]
    for f in forbidden:
        assert f not in sig.parameters, f"Forbidden parameter '{f}' found in adapt_and_predict_image signature!"


def test_label_isolation_runtime_head_weights_frozen():
    # Load joint checkpoint
    model, _ = load_model(JOINT_CKPT, device="cpu", num_classes=100)
    model.eval()

    cls_w_before = model.classifier[1].weight.clone()
    cls_b_before = model.classifier[1].bias.clone()
    rot_w_before = model.rotation_head.weight.clone()
    rot_b_before = model.rotation_head.bias.clone()

    # Create dummy image tensor
    torch.manual_seed(123)
    dummy_img = torch.rand(1, 3, 32, 32)

    # Run adaptation with non-zero lr
    logits, pre_loss, post_loss, rot_corr = adapt_and_predict_image(
        model, dummy_img, ttt_lr=0.01, ttt_steps=3, lambda_rot=1.0
    )

    cls_w_after = model.classifier[1].weight
    cls_b_after = model.classifier[1].bias
    rot_w_after = model.rotation_head.weight
    rot_b_after = model.rotation_head.bias

    # Verify heads have strictly 0.0 change
    assert torch.max(torch.abs(cls_w_before - cls_w_after)).item() == 0.0, "classifier weight modified!"
    assert torch.max(torch.abs(cls_b_before - cls_b_after)).item() == 0.0, "classifier bias modified!"
    assert torch.max(torch.abs(rot_w_before - rot_w_after)).item() == 0.0, "rotation_head weight modified!"
    assert torch.max(torch.abs(rot_b_before - rot_b_after)).item() == 0.0, "rotation_head bias modified!"

    # Verify heads have no residual grad
    assert model.classifier[1].weight.grad is None
    assert model.rotation_head.weight.grad is None
    assert not model.classifier[1].weight.requires_grad
    assert not model.rotation_head.weight.requires_grad


def test_reset_vs_carryover_verification():
    # Load joint checkpoint
    model, _ = load_model(JOINT_CKPT, device="cpu", num_classes=100)
    initial_features_state = copy.deepcopy(model.features.state_dict())

    # Create two distinct test samples
    torch.manual_seed(42)
    sample1 = torch.rand(1, 3, 32, 32)
    sample2 = torch.rand(1, 3, 32, 32)

    # 1. Test standard_ttt: episodic reset
    # Run standard_ttt manually simulating evaluate_condition
    # Sample 1: adapt and reset
    logits1, _, _, _ = adapt_and_predict_image(model, sample1, ttt_lr=0.01, ttt_steps=1)
    # verify features changed during adaptation
    s1_adapted_state = copy.deepcopy(model.features.state_dict())
    diff1 = sum(torch.max(torch.abs(s1_adapted_state[k] - initial_features_state[k])).item() for k in initial_features_state)
    assert diff1 > 0.0, "Features did not adapt on sample 1!"

    # Reset
    model.features.load_state_dict(initial_features_state)

    # Verify reset is bit-for-bit identical to initial joint checkpoint state
    for k in initial_features_state:
        assert torch.equal(model.features.state_dict()[k], initial_features_state[k]), f"Reset failed for key {k}"

    # Sample 2: adapt from reset state
    logits2, _, _, _ = adapt_and_predict_image(model, sample2, ttt_lr=0.01, ttt_steps=1)
    s2_adapted_state = copy.deepcopy(model.features.state_dict())
    diff2 = sum(torch.max(torch.abs(s2_adapted_state[k] - initial_features_state[k])).item() for k in initial_features_state)
    assert diff2 > 0.0, "Features did not adapt on sample 2!"

    # Reset again
    model.features.load_state_dict(initial_features_state)
    for k in initial_features_state:
        assert torch.equal(model.features.state_dict()[k], initial_features_state[k]), f"Reset failed after sample 2 for key {k}"

    # 2. Test online_ttt: carryover accumulation
    # Start fresh from initial_features_state
    model.features.load_state_dict(initial_features_state)

    # Sample 1 adaptation
    adapt_and_predict_image(model, sample1, ttt_lr=0.01, ttt_steps=1)
    state_after_sample1 = copy.deepcopy(model.features.state_dict())

    # Sample 2 adaptation WITHOUT reset (online carryover)
    adapt_and_predict_image(model, sample2, ttt_lr=0.01, ttt_steps=1)
    state_after_sample2 = copy.deepcopy(model.features.state_dict())

    # Verify state after sample 2 is different from initial state AND different from state after sample 1
    diff_s1_init = sum(torch.max(torch.abs(state_after_sample1[k] - initial_features_state[k])).item() for k in initial_features_state)
    diff_s2_s1 = sum(torch.max(torch.abs(state_after_sample2[k] - state_after_sample1[k])).item() for k in initial_features_state)
    diff_s2_init = sum(torch.max(torch.abs(state_after_sample2[k] - initial_features_state[k])).item() for k in initial_features_state)

    assert diff_s1_init > 0.0, "Sample 1 produced no update!"
    assert diff_s2_s1 > 0.0, "Sample 2 produced no update over sample 1 in online mode!"
    assert diff_s2_init > 0.0, "Cumulative update from init is 0!"

    # Verify online adaptation on sample 2 differs from standard_ttt adaptation on sample 2
    # because the starting point for sample 2 was state_after_sample1 rather than initial_features_state
    diff_online_vs_standard_s2 = sum(
        torch.max(torch.abs(state_after_sample2[k] - s2_adapted_state[k])).item() for k in initial_features_state
    )
    assert diff_online_vs_standard_s2 > 0.0, (
        "Online sample 2 state should diverge from standard_ttt sample 2 state due to carryover!"
    )


def test_evaluate_condition_end_to_end_reset_and_carryover():
    # Test with a mock 2-sample DataLoader through evaluate_condition
    model, _ = load_model(JOINT_CKPT, device="cpu", num_classes=100)
    initial_sd = copy.deepcopy(model.features.state_dict())

    torch.manual_seed(999)
    sample_imgs = torch.rand(2, 3, 32, 32)
    sample_lbls = torch.tensor([5, 12], dtype=torch.long)
    sample_sids = ["img_001", "img_002"]

    dataset = torch.utils.data.TensorDataset(sample_imgs, sample_lbls)
    loader = torch.utils.data.DataLoader(
        list(zip(sample_imgs, sample_lbls, sample_sids)),
        batch_size=1,
        shuffle=False,
    )

    # 1. evaluate standard_ttt
    model.features.load_state_dict(initial_sd)
    summary_std, windows_std, logs_std = evaluate_condition(
        model=model,
        loader=loader,
        condition="standard_ttt",
        ttt_lr=0.005,
        ttt_steps=1,
    )
    assert summary_std["update_count"] == 2
    # Verify model features after standard_ttt run was reset to initial_sd
    for k in initial_sd:
        assert torch.equal(model.features.state_dict()[k], initial_sd[k]), (
            f"evaluate_condition for standard_ttt did not reset feature {k} to theta_0"
        )

    # 2. evaluate online_ttt
    model.features.load_state_dict(initial_sd)
    summary_onl, windows_onl, logs_onl = evaluate_condition(
        model=model,
        loader=loader,
        condition="online_ttt",
        ttt_lr=0.005,
        ttt_steps=1,
    )
    assert summary_onl["update_count"] == 2
    # Verify model features after online_ttt accumulated updates and diverge from initial_sd
    diff = sum(torch.max(torch.abs(model.features.state_dict()[k] - initial_sd[k])).item() for k in initial_sd)
    assert diff > 0.0, "online_ttt failed to retain accumulated updates in model.features!"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
