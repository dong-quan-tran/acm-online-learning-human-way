# Week 2 Weekly Report: Transition to Shuffled Single-Pass Online Learning

**Project:** Online Learning – The Human Way  
**Author:** Tuong (Method Explanation & Protocol Review)  
**Date:** October 15, 2026  
**Deliverable:** Week 2 Method Report & Progress Update  

## 1. Executive Summary & Weekly Progress Update

Following Dr. Athitsos's confirmation on October 13, 2026 (*"Yes, this is exactly what I had in mind"*), our project transitioned from Week 1's **class-incremental continual learning** to **shuffled single-pass online learning** on CIFAR-100 (starting with classes 0–19). My work this week focused on documenting the new setting, preserving our Week 1 **Elastic Weight Consolidation (EWC)** work as background, establishing the protocol review invariants, and tracking open method questions.

* **Completed:**
  * Authored the team guide in [docs/online_learning_guide.md](https://github.com/dong-quan-tran/acm-online-learning-human-way/blob/tuong/ewc-notes/docs/online_learning_guide.md) explaining **shuffled single-pass online learning**, the 4-step execution flow, **side-effect-free evaluation**, and the 5 protocol review invariants.
  * Updated [docs/ewc_guide.md](https://github.com/dong-quan-tran/acm-online-learning-human-way/blob/tuong/ewc-notes/docs/ewc_guide.md) with a status note preserving Week 1 **Elastic Weight Consolidation (EWC)** and Fisher Information Matrix estimation as background for possible future work.
  * Reviewed [docs/experiment_protocol.md](https://github.com/dong-quan-tran/acm-online-learning-human-way/blob/tuong/ewc-notes/docs/experiment_protocol.md) and defined the 5 verification checks for auditing `src/data.py`, `src/model.py`, `src/evaluate.py`, `src/train_online.py`, and `configs/online.yaml`.
* **Evidence:**
  * Committed [docs/online_learning_guide.md](https://github.com/dong-quan-tran/acm-online-learning-human-way/blob/tuong/ewc-notes/docs/online_learning_guide.md) (commits `c953503` and `d3b0af8`) and updated [docs/ewc_guide.md](https://github.com/dong-quan-tran/acm-online-learning-human-way/blob/tuong/ewc-notes/docs/ewc_guide.md) (commit `cd41d5b`) on branch `tuong/ewc-notes`.
* **Blocker:**
  * Waiting on final team alignment and pull requests for `src/data.py` (validation split size) and `src/train_online.py` (optimizer and learning rate scheduler settings in `configs/online.yaml`) to complete the code review and run the integrated 20-class experiment.
* **Next Step:**
  * Audit Khoi's and Nguyen's pull requests against the 5 protocol review invariants and collaborate with Alka on `docs/week2_results_summary.md` once `results/online_results.csv` and `results/figures/online_learning_curve.png` are generated.

## 2. Conceptual Shift: What Changed in Week 2

In Week 1, we studied **class-incremental continual learning**, where the model trained on Task 1 (classes 0–9) for 50 epochs and then switched to Task 2 (classes 10–19) for 50 epochs. Because classes 0–9 disappeared completely during Task 2, gradient updates overwrote Task 1 weights, causing **catastrophic forgetting**.

In Week 2, we adopt **shuffled single-pass online learning**:
* **Stationary class distribution:** We combine all selected classes (classes 0–19) upfront and shuffle them once using a recorded `stream_seed`. Every class has an equal probability ($P(Y = k) = 1/20 = 0.05$) of appearing throughout the stream, eliminating artificial task boundaries. Individual **mini-batches** naturally vary in class counts by chance.
* **Single pass (`passes = 1.0`):** The model processes the training stream in **mini-batches** of `batch_size = 32` (including the final smaller **mini-batch** via `drop_last=False`). Every training example is seen **exactly once**.
* **No replay:** After computing cross-entropy loss and applying one gradient update on the current **mini-batch**, that **mini-batch** is immediately discarded. The pipeline maintains **no replay** buffer and never accesses future **mini-batches**. Only model weights and optimizer state carry forward across steps.
* **Learning progress replaces catastrophic forgetting:** Because classes 0–9 never vanish from the stream, Task 1 **catastrophic forgetting** ($F_1 = A_{1,1} - A_{1,2}$) is no longer applicable. The central challenge is **sample efficiency**—how effectively `SmallResNet(num_classes=100)` can learn from a **single pass**. We therefore measure **learning progress** by plotting held-out **validation accuracy** against `examples_processed`.

### Comparison: Multi-Pass Offline Training vs. Shuffled Single-Pass Online Learning

| Dimension | Multi-Pass Offline Training | Shuffled Single-Pass Online Learning |
|---|---|---|
| **Data Access** | Full training split stored and accessed across epochs | Stream of **mini-batches** (`batch_size = 32`); current **mini-batch** only |
| **Number of Passes** | Multiple epochs (e.g., 50 passes; `passes = 50.0`) | Strictly a **single pass** (`passes = 1.0`); each example seen once |
| **Past Data Retention** | Implicit full replay by re-reading the dataset every epoch | **No replay**; each **mini-batch** is discarded immediately after 1 update |
| **Total Gradient Steps** | $\text{Epochs} \times \lceil N_{\text{train}} / 32 \rceil$ updates | Exactly $\lceil N_{\text{train}} / 32 \rceil$ updates |
| **Memory Complexity** | Scales with stored dataset size | $O(1)$ constant memory with respect to stream length |
| **Primary Challenge** | Preventing overfitting across repeated passes | **Sample efficiency** (extracting features from one look per example) |
| **Evaluation Metric** | Final accuracy after multi-epoch convergence | **Learning progress** curve (**validation accuracy** vs. `examples_processed`) |

When comparing **shuffled single-pass online learning** with **multi-pass offline training** in our stretch goals, we report the extra passes, updates, and training time explicitly as a data-access budget comparison rather than an equal-compute comparison. A 1-pass offline run with identical **mini-batches** and seeds serves as an implementation sanity check.

## 3. Online Execution Flow & Side-Effect-Free Evaluation

### 3.1. The 4-Step Execution Flow

Every training step in `src/train_online.py` follows this exact sequence:

```text
Receive labeled batch
→ Update the model once
→ Move to the next batch
→ Check held-out accuracy at recorded intervals
```

1. **Receive labeled batch:** Load a **mini-batch** of `batch_size = 32` images and original CIFAR-100 labels (`0–19`) from the shuffled training stream.
2. **Update the model once:** Run a forward pass through `SmallResNet(num_classes=100)` to produce 100 raw logits per image, compute cross-entropy loss, backpropagate gradients, and execute one optimizer step.
3. **Move to the next batch:** Clear gradients (`optimizer.zero_grad()`) and discard the current **mini-batch** with **no replay**. Retain updated model weights and optimizer state.
4. **Check held-out accuracy at recorded intervals:** Evaluate **validation accuracy** at `0` examples (where unmasked random guessing across 100 logits yields $\approx 1.0\%$ accuracy), roughly every `1,000` `examples_processed` (immediately after the **mini-batch** that crosses the threshold), and at the end of the **single pass** (`passes = 1.0`).

### 3.2. Side-Effect-Free Evaluation Requirements

To guarantee **side-effect-free evaluation** at every checkpoint in `src/evaluate.py`:
* **Disable gradient tracking (`torch.no_grad()`):** Prevents PyTorch from building the autograd computation graph or storing intermediate activations during validation, avoiding memory spikes and ensuring zero gradient updates.
* **Freeze running statistics (`model.eval()`):** Forces Batch Normalization layers to use existing running statistics rather than updating running mean and variance on held-out validation images (which would cause validation data leakage), and disables Dropout.
* **Restore training mode (`model.train()`):** Re-enables gradient tracking and Batch Normalization running-statistic updates before the next training **mini-batch** arrives.

## 4. Protocol Review Invariants & Open Method Questions

### 4.1. The 5 Protocol Review Invariants

All Week 2 code changes in `src/data.py`, `src/model.py`, `src/evaluate.py`, `src/train_online.py`, and `configs/online.yaml` are audited against five invariants:

| # | Invariant | Verification Requirement |
|---|---|---|
| **1** | **No Separate Class Stages** | Combine classes 0–19 before shuffling; remove `TASKS = [1, 2]` sequential loops. |
| **2** | **Fixed Classifier & Global Labels** | Initialize `SmallResNet(num_classes=100)` once from random weights; keep original CIFAR-100 labels (`0–19`); never call `expand_head()` or apply task prediction masks. |
| **3** | **Single Pass & No Replay** | Process every training example once (`passes = 1.0`, `drop_last=False`) with one cross-entropy update per **mini-batch** and **no replay**. If using `OneCycleLR`, set `total_steps = len(train_loader)`. |
| **4** | **Clean Split Separation** | Reserve the validation split using a recorded `split_seed` before shuffling the training stream; verify zero sample overlap across train, validation, and test splits; avoid double normalization between `src/data.py` and `SmallResNet(num_classes=100)`. |
| **5** | **Side-Effect-Free Evaluation** | Wrap evaluation in `model.eval()` and `torch.no_grad()`, restore `model.train()`, exclude evaluation time from `training_seconds`, and log all 14 required columns to `results/online_results.csv`. |

### 4.2. Open Method Questions Tracked

While Dr. Athitsos confirmed the **shuffled single-pass online learning** direction, two team implementation choices require final alignment before locking `configs/online.yaml`:
1. **Validation Split Size:** Finalizing between reserving 100 examples per class (`2,000` validation examples; `8,000` training examples / `250` **mini-batches**) versus a 10% stratified split (`1,000` validation examples; `9,000` training examples / `282` **mini-batches**).
2. **Optimizer & Learning Rate Schedule:** Finalizing between Adam (`learning_rate = 0.001`), SGD with momentum (`0.9`, `learning_rate = 0.01`), or single-pass `OneCycleLR` with `total_steps` matched to the single-pass **mini-batch** count.
