# Online Learning Guide: Shuffled Single-Pass Streaming

**Project:** Online Learning – The Human Way  
**Author:** Tuong (Method Explanation & Protocol Review)  
**Deliverable:** Week 2 Online Learning Guide & Review Reference  
**Context:** CIFAR-100 Shuffled Online Learning (Debugging subset: classes 0–19; Full stage: classes 0–99)

---

## 1. Overview & Purpose

In Week 2, our project transitions from sequential task-based continual learning (Task 1 → Task 2) to **shuffled single-pass online learning**, following Dr. Athitsos's guidance.

Instead of training a model on separate groups of classes one after another for many epochs, we combine all selected classes upfront and feed them to the model in a single continuous stream of small batches. This guide explains the new setting in simple terms, contrasts it with both Week 1's continual-learning setup and traditional multi-pass offline training, and outlines the method checks for our team's implementation.

*(Note: Our Week 1 notes in `docs/ewc_guide.md` are preserved as background reference for potential future extensions, as Elastic Weight Consolidation and Fisher estimation are deferred.)*

---

## 2. How Learning from Incoming Batches Works

In everyday life, humans rarely learn by memorizing an entire textbook fifty times before seeing a new example. Instead, we learn **online**—updating our understanding step by step as new experiences arrive.

In our online-learning setup, the image classifier learns in the exact same way:
1. **Single Fixed Model:** We initialize one neural network (`SmallResNet(num_classes=100)`) from random starting weights and keep the same model and optimizer active throughout the entire run. We use the original CIFAR-100 class IDs (`0–19` in the initial run) without expanding or resetting the classification head.
2. **Stream of Small Batches:** Training images arrive in small labeled mini-batches of **32 examples** (`batch_size = 32`), including the final smaller batch at the end of the stream.
3. **Incremental Updates:** When a batch arrives, the model predicts labels, calculates standard cross-entropy loss on that batch, and performs **one gradient update** to adjust its weights. Model weights and optimizer state carry over smoothly from one batch to the next.
4. **Periodic Checkpoints:** At recorded intervals (before training starts, roughly every 1,000 training examples processed, and at the end of the stream), we pause training to measure accuracy on a separate held-out validation set.

### Online Learning Execution Flow

```text
Receive labeled batch
→ Update the model once
→ Move to the next batch
→ Check held-out accuracy at recorded intervals
```

During the evaluation step (`Check held-out accuracy at recorded intervals`), gradients are disabled (`torch.no_grad()`) and the network switches to evaluation mode (`model.eval()`) so that checking accuracy never alters the model's weights or BatchNorm running statistics. Afterward, the model returns to training mode (`model.train()`) and processes the next incoming batch.

---

## 3. Why Classes Are Mixed Instead of Separated into Tasks

In Week 1, we divided CIFAR-100 into strict class blocks: the model trained exclusively on **Task 1 (classes 0–9)** and then switched exclusively to **Task 2 (classes 10–19)**. That setup created an abrupt, artificial change in the data distribution at the task boundary.

In Week 2, we **combine all selected classes (classes 0–19) before shuffling**:
- **Fixed Underlying Distribution:** Every selected class has an equal chance of appearing throughout the entire stream, from the very first batch to the final batch.
- **Natural Batch-to-Batch Variation:** Because the combined dataset is randomly shuffled once with a recorded seed (`stream_seed`), individual batches of 32 images will naturally contain varying numbers of examples per class by chance—and a single batch does not need to contain all 20 classes.
- **Focus on Continuous Acquisition:** Mixing classes removes artificial stage transitions so we can directly study how a neural network builds representations when all target classes coexist in an incoming stream.
- **Practical Simulation Limit:** Because CIFAR-100 is a finite dataset, shuffling the training split once simulates a stationary data stream; we note in our analysis that it is a finite without-replacement stream rather than an infinite independent stream.

---

## 4. What "One Pass" and "No Replay" Mean

Two strict constraints define our Week 2 online baseline:

### 4.1. One Pass (Single-Pass Streaming)
- **What it means:** The model sees each training image in the stream **exactly once**.
- **In practice:** If our 20-class training split has $N$ images (after setting aside validation data), the training loop runs for $\lceil N / 32 \rceil$ batches—including the final partial batch if $N$ is not divisible by 32—and then stops (`passes = 1.0`).
- **Why it matters:** In true streaming scenarios where data arrives continuously, a system must extract useful signal from each batch on the fly rather than cycling through the entire dataset for 50 epochs.

### 4.2. No Replay (Exemplar-Free Updates)
- **What it means:** Once the model finishes its single gradient update on the current batch, that batch is **discarded**.
- **In practice:** The training script must not store old images in an episodic memory buffer to mix into later batches, and it must never look ahead at future batches in the stream. Only the model's learned parameters and optimizer state are retained between steps.
- **Why it matters:** Operating without replay tests the raw ability of standard gradient descent and our network architecture to accumulate knowledge solely through parameter updates, with minimal memory overhead.

---

## 5. How Online Learning Differs from Repeated Offline Training

Traditional deep learning uses **multi-pass offline training** (batch learning), where the entire training set is stored upfront and the optimizer loops over all examples across many epochs.

| Feature | Shuffled Online Learning (Week 2 Main Run) | Multi-Pass Offline Training (Stretch Comparison) |
|---|---|---|
| **Data Access** | Sequential stream of batches; current batch only | Full training dataset remains accessible at all times |
| **Number of Passes** | **1 pass** (`passes = 1.0`); each example seen once | **Multiple passes** (e.g., 10–50 epochs over the dataset) |
| **Updates per Run** | $\lceil N_{\text{train}} / 32 \rceil$ total gradient steps | $\text{Epochs} \times \lceil N_{\text{train}} / 32 \rceil$ total gradient steps |
| **Memory of Past Data** | None (batches are discarded immediately after 1 update) | Implicit full replay via repeated passes every epoch |
| **Primary Trade-off** | Fast, lightweight, single-pass, lower final accuracy | Higher compute and data-access cost, higher final accuracy |

### Clarifying Our Online vs. Offline Comparison
Once our 20-class single-pass online run is complete, our stretch goal compares it against multi-pass offline training:
1. **Controlled Setup:** Both runs must use the exact same `SmallResNet(num_classes=100)` model initialization, train/validation/test splits, and image transforms.
2. **Access-Budget Comparison (Not Equal-Budget):** Multi-pass offline training performs many more gradient updates and sees every image multiple times. When reporting results, we must explicitly report the extra passes, update count, and training time rather than framing it as an equal-training-budget comparison.
3. **One-Pass Sanity Check:** Running the offline trainer for **1 epoch** with the exact same seed, batch order, and optimizer settings should produce results identical (or nearly identical) to our single-pass online run. This serves as a clean verification check for our codebase.

---

## 6. Why Learning Progress Replaces Forgetting as the Main Focus

In Week 1, our primary metric was **catastrophic forgetting** ($F_1 = A_{1,1} - A_{1,2}$) because classes 0–9 vanished completely when Task 2 began, causing Task 2 gradients to overwrite Task 1 weights.

In Week 2's shuffled online setting, **learning progress** replaces forgetting as the main focus for three reasons:
1. **No Missing Class Stages:** Because classes 0–19 are mixed throughout the entire stream, classes 0–9 never stop appearing. The model continues to receive gradient signal for all 20 classes from start to finish.
2. **Sample Efficiency Under Single-Pass Constraints:** When a model only gets a single look at each training image (`1 pass`), the central challenge is no longer preventing an old task stage from being erased—it is **how quickly the model can learn accurate features from limited updates**.
3. **Tracking the Learning Trajectory:** Rather than comparing two end-of-task bar charts, we measure how held-out validation accuracy climbs as a function of **`examples_processed`** (0 examples, ~1,000 examples, ~2,000 examples, ..., up to the end of the single pass). This produces a continuous **online learning curve** (`results/figures/online_learning_curve.png`).

---

## 7. Protocol & Implementation Review Checklist

Before merging our Week 2 code and running the final 20-class baseline, we verify five core methodological invariants across `src/data.py`, `src/model.py`, `src/evaluate.py`, `src/train_online.py`, and `configs/online.yaml`:

1. **No Separate Class-Based Stages:**
   - Classes 0–19 are filtered and combined *before* shuffling.
   - Sequential task loops (`TASKS = [1, 2]`) are removed from the training pipeline.
2. **Fixed Classifier & Global Labels:**
   - `SmallResNet(num_classes=100)` is initialized once from random weights.
   - Original CIFAR-100 fine labels (`0–19`) are preserved without remapping or calling `expand_head()`.
   - Model and optimizer states persist across batches without restarting.
3. **Strict Single Pass Without Replay:**
   - Every selected training example is processed exactly once (`batch_size = 32`), including the final smaller batch (`drop_last=False`).
   - Exactly one cross-entropy update occurs per batch, with no replay buffer or future-batch lookahead.
   - If `OneCycleLR` is used, its `total_steps` equals the number of batches in **one pass**, not 50 epochs.
4. **Clean Split Separation & Transform Consistency:**
   - Validation examples are split from the training set using a recorded seed *before* building the training stream.
   - Training, validation, and test sets have zero overlapping sample identities.
   - Image normalization is applied once (since Nguyen's `SmallResNet` normalizes inputs internally, `src/data.py` must not double-normalize).
5. **Side-Effect-Free Evaluation & Shared CSV Logging:**
   - Validation evaluation runs at 0 examples, roughly every 1,000 examples (evaluating right after the batch that crosses the threshold and logging the exact `examples_processed`), and at stream completion.
   - Evaluation uses `model.eval()` and `torch.no_grad()`, leaving weights and BatchNorm running statistics completely unchanged, and restores `model.train()` afterward.
   - Checkpoint rows in `results/online_results.csv` follow the exact 14-column schema (`mode,run_id,model_seed,stream_seed,num_classes,examples_processed,updates,passes,batch_size,learning_rate,optimizer,eval_split,accuracy,training_seconds`), with `training_seconds` excluding evaluation time.

---

## 8. Open Method Questions to Track

As noted in `docs/online_learning_direction.md` and `docs/experiment_protocol.md`, while Dr. Athitsos suggested the fixed-distribution, random-order online setting, the following specific implementation choices are team defaults awaiting his final confirmation:
- **Exact validation split size:** Percentage/count of training examples reserved per class (or globally across classes 0–19) before forming the stream.
- **Optimizer & scheduler choice:** Final choice of optimizer (e.g., SGD vs. Adam/AdamW), base learning rate, and whether to use a constant learning rate or single-pass `OneCycleLR` in `configs/online.yaml`.
- **Offline comparison epoch budget:** Exact number of passes/epochs to use when running the stretch-goal multi-pass offline comparison.
