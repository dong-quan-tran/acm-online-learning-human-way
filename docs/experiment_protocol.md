# Experiment Protocol

## Research Question

How effectively can a neural network learn from a shuffled stream of CIFAR-100 images in a single pass, and how does its accuracy compare with offline training?

## Initial Experiment

- Dataset: CIFAR-100 (fine labels)
- Debugging stage: classes 0–19 mixed together; main stage: all 100 classes
- Model: one small CNN initialized from scratch with 100 output classes
- Learning order: shuffle training examples once, then process each batch sequentially
- Online baseline: batch size 32, one gradient update per batch, one pass only, including the final partial batch
- Loss: cross-entropy; no replay, future-data access, or EWC
- Data splits: separate validation data before training; reserve official test data for final evaluation
- Initial seed: 42; record model and stream seeds separately
- Evaluation: held-out accuracy every 1,000 processed examples and at the end

## Required Measurements

- Held-out accuracy at each checkpoint (decimal between 0 and 1)
- Number of examples processed and gradient updates
- Final accuracy, training time, and memory usage (with measurement method)
- Primary plot: accuracy versus examples processed

## Fair Comparison Rules

Compare online and multi-pass offline training using the same model, data splits, and preprocessing. Record differences in passes, updates, and runtime; do not claim equal compute. Agree on optimizer, learning rate, transforms, and offline epoch budget before comparisons. Repeat final runs with three recorded seeds.

## Reproducibility Rules

Record Python and package versions, device, seeds, configuration, data splits, and exact run commands. Verify each training example is used once, stream order is reproducible, and evaluation does not update model weights or statistics. Do not upload datasets, checkpoints, secrets, or `.venv` folders.

## Previous Research

Class-incremental learning, catastrophic forgetting, and EWC are preserved as background research and deferred extensions. This protocol is proposed and pending final team confirmation.
