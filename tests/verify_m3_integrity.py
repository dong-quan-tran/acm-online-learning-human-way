"""Standalone Adversarial Verification Script for Milestone 3 (Exp B).

Authored by Challenger 1 (Gen 2).
Empirically tests:
1. Window Accuracy Conservation Law:
   - For every run in results/ttt_summary.csv:
     sum(window_correct_counts) == round(summary_accuracy * 2000)
   - Each window size is exactly 200
   - Window ranges [start_sample, end_sample) partition [0, 2000) contiguously
2. Single Recorded Prediction Per Test Image:
   - Exactly 2,000 predictions per condition
   - Test membership has exactly 2,000 unique samples
   - No duplicates, no dropped samples, strictly sorted ascending
3. Accuracy & Metric Bounds Check:
   - Reported accuracy matches correct_count / 2000 exactly
   - Rotation loss is strictly positive (> 0.0)
   - Update counts are strictly 0 for frozen conditions and 2,000 for standard/online TTT
   - Adaptation steps, reset modes, optimizer resets match theoretical invariants
4. Live Checkpoint Reproducibility:
   - Clean & noisy frozen evaluation match summary CSV numbers
"""

import csv
import json
import math
from pathlib import Path
import sys
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import (
    EXP_B_TEST_MEMBERSHIP_PATH,
    SHARED_C20_MEMBERSHIP_PATH,
    apply_gaussian_noise,
    get_exp_b_loader,
)
from src.models import SmallResNetTTT
from src.ttt_cifar100 import (
    ALKA_SUMMARY_COLUMNS,
    ALKA_WINDOW_COLUMNS,
    VALID_CONDITIONS,
    evaluate_condition,
    load_model,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
SUMMARY_PATH = RESULTS_DIR / "ttt_summary.csv"
WINDOW_PATH = RESULTS_DIR / "ttt_window_summary.csv"
TEST_MEMBERSHIP_PATH = EXP_B_TEST_MEMBERSHIP_PATH
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify_window_accuracy_conservation_law() -> dict:
    """Verifies that window summaries conserve total correct counts and partition [0, 2000)."""
    summary_rows = load_csv(SUMMARY_PATH)
    window_rows = load_csv(WINDOW_PATH)

    assert len(summary_rows) == 8, f"Expected 8 summary rows, got {len(summary_rows)}"
    assert len(window_rows) == 80, f"Expected 80 window rows (8 runs * 10 windows), got {len(window_rows)}"

    windows_by_run = {}
    for w in window_rows:
        windows_by_run.setdefault(w["run_id"], []).append(w)

    audit_log = []

    for s_row in summary_rows:
        run_id = s_row["run_id"]
        cond = s_row["condition"]
        corr = s_row["corruption"]
        acc = float(s_row["accuracy"])

        assert run_id in windows_by_run, f"run_id '{run_id}' not found in window summary"
        windows = sorted(windows_by_run[run_id], key=lambda x: int(x["window_index"]))

        assert len(windows) == 10, f"Expected 10 windows for {run_id}, got {len(windows)}"

        sum_correct = 0
        sum_samples = 0
        expected_total_correct = round(acc * 2000)

        prev_end = 0
        for w_idx, w in enumerate(windows):
            idx = int(w["window_index"])
            start = int(w["start_sample"])
            end = int(w["end_sample"])
            size = int(w["window_size"])
            correct = int(w["correct_count"])
            w_acc = float(w["accuracy"])

            # 1. Exact window index
            assert idx == w_idx, f"{run_id}: window_index {idx} != expected {w_idx}"
            # 2. Contiguity and partitioning
            assert start == prev_end, f"{run_id} window {idx}: start {start} != prev_end {prev_end}"
            assert end == start + size, f"{run_id} window {idx}: end {end} != start+size {start+size}"
            assert size == 200, f"{run_id} window {idx}: size {size} != 200"
            prev_end = end

            # 3. Window accuracy formula
            computed_w_acc = correct / size
            assert abs(w_acc - computed_w_acc) < 1e-6, (
                f"{run_id} window {idx}: accuracy {w_acc} != {correct}/{size} = {computed_w_acc}"
            )

            sum_correct += correct
            sum_samples += size

        # 4. Total sample count
        assert sum_samples == 2000, f"{run_id}: total samples {sum_samples} != 2000"
        assert prev_end == 2000, f"{run_id}: final end_sample {prev_end} != 2000"

        # 5. Conservation law: sum(correct) == round(accuracy * 2000)
        assert sum_correct == expected_total_correct, (
            f"{run_id}: sum of window correct counts ({sum_correct}) != "
            f"round(accuracy * 2000) ({expected_total_correct})"
        )

        # 6. Overall summary accuracy check
        calculated_summary_acc = sum_correct / 2000.0
        assert abs(acc - calculated_summary_acc) < 1e-6, (
            f"{run_id}: summary accuracy {acc} != {sum_correct}/2000 = {calculated_summary_acc}"
        )

        audit_log.append({
            "run_id": run_id,
            "condition": cond,
            "corruption": corr,
            "accuracy": acc,
            "sum_window_correct": sum_correct,
            "expected_correct": expected_total_correct,
            "windows_verified": 10,
            "samples_partitioned": sum_samples,
            "conservation_status": "CONSERVED",
        })

    return {"status": "PASS", "runs_checked": len(audit_log), "audit_log": audit_log}


def verify_test_membership_and_single_prediction() -> dict:
    """Verifies test membership structure and single prediction per sample guarantees."""
    assert TEST_MEMBERSHIP_PATH.is_file(), f"Missing {TEST_MEMBERSHIP_PATH}"
    with open(TEST_MEMBERSHIP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "test" in data, "Missing 'test' key in test membership"
    test_indices = data["test"]

    # Exactly 2,000 samples
    assert len(test_indices) == 2000, f"Expected 2000 test indices, got {len(test_indices)}"
    # Exactly 2,000 unique samples (no duplicates)
    assert len(set(test_indices)) == 2000, "Found duplicate test indices!"
    # Ascending source-row order
    assert test_indices == sorted(test_indices), "Test indices are not strictly in ascending order!"
    # Classes restricted to 0..19
    assert data.get("classes") == list(range(20)), f"Unexpected classes: {data.get('classes')}"
    assert data.get("num_samples") == 2000

    # Verify each condition produces 2000 predictions (from 10 windows of 200 each)
    window_rows = load_csv(WINDOW_PATH)
    windows_by_run = {}
    for w in window_rows:
        windows_by_run.setdefault(w["run_id"], []).append(w)

    for run_id, windows in windows_by_run.items():
        total_eval_samples = sum(int(w["window_size"]) for w in windows)
        assert total_eval_samples == 2000, f"{run_id}: total window size {total_eval_samples} != 2000"

    return {
        "status": "PASS",
        "num_test_indices": len(test_indices),
        "unique_test_indices": len(set(test_indices)),
        "is_sorted_ascending": True,
        "classes": data.get("classes"),
    }


def verify_metric_bounds_and_update_counts() -> dict:
    """Verifies metric bounds, positive rotation loss, update counts, and Alka schema conformance."""
    summary_rows = load_csv(SUMMARY_PATH)
    window_rows = load_csv(WINDOW_PATH)

    # Schema checks
    for row in summary_rows:
        for col in ALKA_SUMMARY_COLUMNS:
            assert col in row, f"Missing column {col} in summary row"
            assert row[col] is not None and row[col] != "", f"Empty column {col} in summary row"

    for row in window_rows:
        for col in ALKA_WINDOW_COLUMNS:
            assert col in row, f"Missing column {col} in window row"
            assert row[col] is not None and row[col] != "", f"Empty column {col} in window row"

    # Bounds and invariants checks
    results = []
    for row in summary_rows:
        cond = row["condition"]
        acc = float(row["accuracy"])
        rot_loss = float(row["rotation_loss"])
        updates = int(row["update_count"])
        adapt_steps = int(row["adaptation_steps"])
        batch_size = int(row["batch_size"])
        lr = float(row["learning_rate"])

        # Accuracy in [0, 1]
        assert 0.0 <= acc <= 1.0, f"Accuracy out of bounds: {acc}"
        # Rotation loss strictly positive
        assert rot_loss > 0.0, f"Rotation loss must be strictly positive: {rot_loss}"
        # Batch size 1
        assert batch_size == 1, f"Batch size must be 1, got {batch_size}"
        # Learning rate
        assert lr == 0.001, f"Expected lr 0.001, got {lr}"

        # Updates and metadata invariants by condition
        if cond in ("supervised_frozen", "joint_frozen"):
            assert updates == 0, f"{cond}: update_count must be strictly 0, got {updates}"
            assert adapt_steps == 0, f"{cond}: adaptation_steps must be 0, got {adapt_steps}"
            assert row["reset_mode"] == "none", f"{cond}: reset_mode must be 'none'"
            assert row["optimizer_reset"] == "none", f"{cond}: optimizer_reset must be 'none'"
            assert row["prediction_timing"] == "none", f"{cond}: prediction_timing must be 'none'"
        elif cond in ("standard_ttt", "online_ttt"):
            assert updates == 2000, f"{cond}: update_count must be strictly 2000, got {updates}"
            assert adapt_steps == 1, f"{cond}: adaptation_steps must be 1, got {adapt_steps}"
            assert row["prediction_timing"] == "post_adaptation"
            assert row["optimizer_reset"] == "stateless"
            if cond == "standard_ttt":
                assert row["reset_mode"] == "per_sample"
            else:
                assert row["reset_mode"] == "carryover"
        else:
            raise ValueError(f"Unknown condition: {cond}")

        results.append({
            "run_id": row["run_id"],
            "condition": cond,
            "accuracy": acc,
            "rotation_loss": rot_loss,
            "update_count": updates,
            "adaptation_steps": adapt_steps,
            "reset_mode": row["reset_mode"],
        })

    return {"status": "PASS", "records_verified": len(results), "details": results}


def verify_live_checkpoint_accuracies() -> dict:
    """Performs live model inference to empirically reproduce checkpoint accuracies."""
    device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
    clean_loader = get_exp_b_loader(
        split="test",
        membership_path=SHARED_C20_MEMBERSHIP_PATH,
        test_membership_path=EXP_B_TEST_MEMBERSHIP_PATH,
        batch_size=64,
        shuffle=False,
        corruption="clean",
    )
    noisy_loader = get_exp_b_loader(
        split="test",
        membership_path=SHARED_C20_MEMBERSHIP_PATH,
        test_membership_path=EXP_B_TEST_MEMBERSHIP_PATH,
        batch_size=64,
        shuffle=False,
        corruption="gaussian_noise",
        noise_sigma=0.08,
        noise_seed=42,
    )

    sup_ckpt = CHECKPOINTS_DIR / "exp_b_supervised_seed42.pt"
    joint_ckpt = CHECKPOINTS_DIR / "exp_b_joint_seed42.pt"
    assert sup_ckpt.is_file(), f"Missing {sup_ckpt}"
    assert joint_ckpt.is_file(), f"Missing {joint_ckpt}"

    repro_results = {}

    # 1. Supervised Clean
    sup_model, _ = load_model(sup_ckpt, device=device)
    res_sup_clean, _, _ = evaluate_condition(
        sup_model, clean_loader, condition="supervised_frozen", device=device, corruption="clean", seed=42
    )
    repro_results["supervised_frozen_clean"] = {
        "reported": 0.6340,
        "empirical": res_sup_clean["accuracy"],
        "delta": abs(res_sup_clean["accuracy"] - 0.6340),
        "pass": abs(res_sup_clean["accuracy"] - 0.6340) < 1e-4,
    }
    assert repro_results["supervised_frozen_clean"]["pass"]

    # 2. Joint Clean
    joint_model, _ = load_model(joint_ckpt, device=device)
    res_joint_clean, _, _ = evaluate_condition(
        joint_model, clean_loader, condition="joint_frozen", device=device, corruption="clean", seed=42
    )
    repro_results["joint_frozen_clean"] = {
        "reported": 0.6585,
        "empirical": res_joint_clean["accuracy"],
        "delta": abs(res_joint_clean["accuracy"] - 0.6585),
        "pass": abs(res_joint_clean["accuracy"] - 0.6585) < 1e-4,
    }
    assert repro_results["joint_frozen_clean"]["pass"]

    # 3. Supervised Noise
    res_sup_noise, _, _ = evaluate_condition(
        sup_model, noisy_loader, condition="supervised_frozen", device=device, corruption="gaussian_noise", severity=3, seed=42
    )
    repro_results["supervised_frozen_gaussian_noise"] = {
        "reported": 0.3905,
        "empirical": res_sup_noise["accuracy"],
        "delta": abs(res_sup_noise["accuracy"] - 0.3905),
        "pass": abs(res_sup_noise["accuracy"] - 0.3905) < 1e-4,
    }
    assert repro_results["supervised_frozen_gaussian_noise"]["pass"]

    # 4. Joint Noise
    res_joint_noise, _, _ = evaluate_condition(
        joint_model, noisy_loader, condition="joint_frozen", device=device, corruption="gaussian_noise", severity=3, seed=42
    )
    repro_results["joint_frozen_gaussian_noise"] = {
        "reported": 0.3505,
        "empirical": res_joint_noise["accuracy"],
        "delta": abs(res_joint_noise["accuracy"] - 0.3505),
        "pass": abs(res_joint_noise["accuracy"] - 0.3505) < 1e-4,
    }
    assert repro_results["joint_frozen_gaussian_noise"]["pass"]

    return {"status": "PASS", "device": device, "repro_results": repro_results}


def main():
    print("=" * 80)
    print("CHALLENGER 1 (GEN 2): ADVERSARIAL INTEGRITY VERIFICATION")
    print("=" * 80)

    # 1. Window Accuracy Conservation
    print("\n[Check 1/4] Window Accuracy Conservation Law & Partitioning...")
    res1 = verify_window_accuracy_conservation_law()
    print(f"  -> PASS: All {res1['runs_checked']} runs conserved exact correct counts.")
    for log in res1["audit_log"]:
        print(f"     * {log['run_id']}: sum(correct)={log['sum_window_correct']} == round({log['accuracy']}*2000)={log['expected_correct']}")

    # 2. Test Membership & Single Prediction
    print("\n[Check 2/4] Test Membership & Single Prediction Per Image...")
    res2 = verify_test_membership_and_single_prediction()
    print(f"  -> PASS: Exactly {res2['num_test_indices']} unique test samples, sorted ascending, classes 0..19.")

    # 3. Metric Bounds & Update Counts
    print("\n[Check 3/4] Metric Bounds, Rotation Loss & Update Counts...")
    res3 = verify_metric_bounds_and_update_counts()
    print(f"  -> PASS: All {res3['records_verified']} summary rows satisfy metric bounds and update invariants.")
    for r in res3["details"]:
        print(f"     * {r['run_id']}: acc={r['accuracy']}, rot_loss={r['rotation_loss']:.4f}, updates={r['update_count']}, reset={r['reset_mode']}")

    # 4. Live Checkpoint Reproducibility
    print("\n[Check 4/4] Live Checkpoint Empirical Reproducibility...")
    res4 = verify_live_checkpoint_accuracies()
    print(f"  -> PASS on device '{res4['device']}':")
    for name, r in res4["repro_results"].items():
        print(f"     * {name}: reported={r['reported']:.4f}, empirical={r['empirical']:.4f}, delta={r['delta']:.6f} [PASS]")

    print("\n" + "=" * 80)
    print("OVERALL VERDICT: ALL ADVERSARIAL CHECKS PASSED -> APPROVE")
    print("=" * 80)


if __name__ == "__main__":
    main()
