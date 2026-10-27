# Current Direction: Shuffled Online Learning

Updated: October 13, 2026

## Why We Are Changing

Dr. Athitsos recommended returning the main focus to online learning. His email described a simpler setting where the underlying class distribution stays fixed and classes appear in random order. On October 13, he confirmed our understanding: "Yes, this is exactly what I had in mind." The agreed direction is mixed classes, small incoming batches, one pass without replay, and accuracy tracked as data arrives. Specific implementation settings below remain team choices.

## Before and After

Before: train classes 0–9, then classes 10–19, and measure forgetting.

Now: mix examples from a fixed class set, update the model as batches arrive, and measure how learning improves over time.

This is not class-incremental training: we are not introducing new class groups in separate stages. Individual batches may contain different class counts by chance. A shuffled finite dataset simulates a stationary stream; it is not an exactly independent infinite stream.

## Proposed Protocol

- Dataset: CIFAR-100, using fine labels.
- Debugging stage: classes 0–19 mixed together.
- Main stage: all 100 classes, after the debugging pipeline works.
- Model: one small CNN, initialized from scratch; retain the planned 100-output head and global labels in both stages.
- Stream: shuffle the combined training subset once with a recorded seed.
- Updates: batch size 32, one gradient update per incoming batch, one pass only; keep the final partial batch.
- Loss: standard cross-entropy; no EWC penalty.
- Access: current batch only; no future-batch access and no replay buffer. Discard each batch after its update. Stored model and optimizer state are allowed.
- Splits: reserve validation examples from training data before forming the stream. Do not use official test examples for training or tuning. Record split membership and preprocessing.
- Seeds: record model initialization and stream-order seeds. Start with seed 42; repeat final runs with three recorded seeds.
- Evaluation: record held-out accuracy at fixed intervals, initially every 1,000 processed training examples and at the end. Evaluation must not update weights or model statistics. Use validation curves during development and the official test set for the frozen final protocol.
- Configuration: agree and record optimizer, learning rate, transforms, evaluation intervals, and offline epoch budget before comparisons.

These choices are project defaults, not universal requirements for online learning.

## Results and Comparison

Primary output: accuracy versus number of training examples processed. Also record final accuracy, elapsed training time, update count, and memory use with its measurement method.

Proposed checkpoint CSV fields:

```csv
mode,run_id,model_seed,stream_seed,num_classes,examples_processed,updates,passes,batch_size,learning_rate,optimizer,eval_split,accuracy,training_seconds
```

Accuracy uses decimals from 0 to 1. One row represents one evaluation checkpoint. Each run must link to its saved configuration.

First complete the online baseline. Then compare with multi-pass offline training using the same model, splits, and preprocessing. Report the extra passes, updates, and time explicitly; this is an access-budget comparison, not an equal-compute claim. A one-epoch offline run using identical batches and updates is a useful implementation sanity check.

Predict-before-update accuracy is optional and must be labeled separately from held-out accuracy.

## Acceptance Checks

- [ ] Classes are combined before shuffling; no class-blocked task boundaries.
- [ ] Selected training samples appear exactly once per run, including the final partial batch.
- [ ] No training/validation/test overlap.
- [ ] Global labels and expected image tensors are verified.
- [ ] Same seed reproduces stream order; different seeds change it.
- [ ] Examples processed and update counts match the stream.
- [ ] No replay or future-data access in the online baseline.
- [ ] Evaluation does not change model parameters or running statistics.
- [ ] Plotter and trainer use the same CSV schema and accuracy units.
- [ ] A fresh checkout can reproduce the small run using documented commands.

## Team Transition

Tuong: update the method explanation and review this protocol. Preserve EWC notes as background.

Khoi: adapt the loader to one mixed stream and add coverage, ordering, label, and split tests. Preserve useful existing subset functions.

Nguyen: implement the CNN, single-pass training loop, checkpoints, and held-out evaluation. Coordinate the loader interface with Khoi.

Alka: align CSV outputs and learning-curve plots; update the older protocol and checklists to match this direction.

Dong Quan Tran: record advisor guidance, coordinate interfaces and reviews, and track effort and follow-through rather than accuracy alone.

Do not delete existing contributions. EWC, Fisher estimation, forgetting metrics, and class-blocked streams are deferred extensions. Older task-based documents are historical and must not override this current plan. No implementation, runtime verification, or passing tests are implied by this documentation update.
