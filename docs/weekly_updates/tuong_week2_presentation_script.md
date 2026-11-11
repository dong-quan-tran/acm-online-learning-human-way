# Week 2 Presentation Script: Shuffled Single-Pass Online Learning

**Presenter:** Tuong (Method Explanation & Protocol Review)  
**Slides:** [Tuong - Week 2 Progress Report: Shuffled Single-Pass Online Learning](https://docs.google.com/presentation/d/1kYDXKmdIv0HP23V1YP8KWxJ_mrAN9oEyn38faDCo7BM/edit)  
**Shared Drive Folder:** [Week 2](https://drive.google.com/drive/folders/1DShhyW7QlneW08mWTDpXQ779gl26riNM)  

## Slide 1: Week 2 Progress — Shuffled Single-Pass Online Learning

* **On-Slide Visuals:**
  * **Week 1 (Deferred):** Class-Incremental Continual Learning (`Classes 0–9 → 50 Epochs → Classes 10–19`, Metric: Catastrophic Forgetting, EWC Guide preserved as background)
  * **Week 2 (Active):** Shuffled Single-Pass Online Learning (`Classes 0–19 Mixed Upfront, P = 1/20`, `Single Pass (passes = 1.0) & No Replay`, Metric: Learning Progress)
* **Speaker Script (~45 seconds):**
  > "Hi everyone, here is my Week 2 update on our method transition and protocol review. Following Dr. Athitsos's confirmation on October 13, we shifted from Week 1's **class-incremental continual learning** to **shuffled single-pass online learning** on CIFAR-100. In Week 1, we trained classes 0–9 for 50 epochs and then switched to classes 10–19, which caused **catastrophic forgetting**. This week, I preserved our **Elastic Weight Consolidation (EWC)** guide in `docs/ewc_guide.md` as background for future work, and authored our new method guide in `docs/online_learning_guide.md`. In our new setting, we pool classes 0–19 upfront into a **stationary class distribution** ($P = 1/20$), train for a **single pass** (`passes = 1.0`) with **no replay**, and track **learning progress** instead of **catastrophic forgetting**."

## Slide 2: Single-Pass Online Stream vs. Multi-Pass Offline

* **On-Slide Visuals:**
  * **Multi-Pass Offline Training:** `50 Passes` | `Full Dataset Stored` | `High Compute`
  * **Shuffled Single-Pass Online Learning:** `1 Pass (passes = 1.0)` | `Mini-Batch (B = 32)` | `No Replay` | `O(1) Memory` | `Challenge: Sample Efficiency`
  * **4-Step Loop:** `1. Receive Labeled Mini-Batch (B = 32)` → `2. Update Model Once (SmallResNet-100)` → `3. Discard Mini-Batch (No Replay)` → `4. Check Validation Accuracy (~Every 1,000 Examples)`
* **Speaker Script (~60 seconds):**
  > "Slide 2 contrasts **multi-pass offline training** with **shuffled single-pass online learning**. In **multi-pass offline training**, the entire dataset stays on disk and the model loops over every image 50 times. In **shuffled single-pass online learning**, data arrives as a stream of **mini-batches** of `batch_size = 32`. As shown in the 4-step flow at the bottom, the model receives one labeled **mini-batch**, performs one cross-entropy gradient step on `SmallResNet(num_classes=100)`, immediately discards that **mini-batch** with **no replay** buffer, and checks held-out **validation accuracy** roughly every 1,000 `examples_processed`. Because the model only gets one look at each example (`passes = 1.0`), memory stays constant ($O(1)$), and the primary challenge becomes **sample efficiency**."

## Slide 3: Side-Effect-Free Evaluation Protocol

* **On-Slide Visuals:**
  * `1. model.eval()` → Freeze BatchNorm Running Stats (No Data Leakage)
  * `2. torch.no_grad()` → Disable Autograd Graph ($O(1)$ Eval Memory)
  * `3. model.train()` → Restore Training State Immediately
  * **Key Callout:** **Validation Accuracy** vs. `examples_processed` (`Unmasked 100-Output Head → 1.0% Random Baseline at 0 Examples`)
* **Speaker Script (~45 seconds):**
  > "Slide 3 covers our **side-effect-free evaluation** protocol. Because we pause the stream every 1,000 `examples_processed` to measure **validation accuracy**, we have to make sure evaluation never corrupts training. First, `model.eval()` freezes Batch Normalization running statistics so held-out validation images never leak into the model's running mean and variance. Second, `torch.no_grad()` disables PyTorch's autograd graph so we don't store intermediate activations or run out of GPU memory. Third, we immediately call `model.train()` before the next **mini-batch** arrives. Also note at the bottom: because we keep an unmasked 100-output head on `SmallResNet(num_classes=100)`, random guessing at 0 examples starts at roughly 1.0% **validation accuracy**, not 5.0%."

## Slide 4: 5 Protocol Invariants & Next Steps

* **On-Slide Visuals:**
  * **5 Invariants:** `1. No Separate Class Stages` | `2. Fixed Classifier (SmallResNet-100)` | `3. Single Pass & No Replay` | `4. Clean Split Separation` | `5. Side-Effect-Free Evaluation`
  * **Status:** Completed `docs/online_learning_guide.md` (`c953503`) & `docs/ewc_guide.md` (`cd41d5b`) | Open: Validation Split (`2,000` vs. `1,000`) & Optimizer/Scheduler (`configs/online.yaml`) | Next: Audit `src/data.py` & `src/train_online.py` PRs + Results Summary
* **Speaker Script (~45 seconds):**
  > "Finally, Slide 4 summarizes the 5 protocol review invariants I established for auditing our Week 2 pull requests: combining classes 0–19 upfront with no task stages, keeping a single fixed `SmallResNet(num_classes=100)` with original labels, enforcing a **single pass** (`passes = 1.0`, `drop_last=False`) with **no replay**, isolating splits before shuffling without double-normalizing images, and enforcing **side-effect-free evaluation**. My updates to `docs/online_learning_guide.md` and `docs/ewc_guide.md` are complete on branch `tuong/ewc-notes`. Next, once we lock our final validation split size and optimizer settings in `configs/online.yaml`, I will review Khoi's and Nguyen's PRs against these 5 invariants and help Alka summarize the 20-class learning curve."
