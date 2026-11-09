# MNIST optimizer alignment with Nguyen's CIFAR run

New configuration: `configs/mnist_nguyen_optimizer.yaml`. The existing
`configs/mnist_9000.yaml`, its 96.4% result, and the full-data result are preserved.

Reference inspected: `src/rawsgd-cifar-100.ipynb` at origin/main `3d356f9`.
Its CFG uses AdamW, max learning rate 0.0005, weight decay 0.0005,
OneCycleLR with pct_start 0.15, dropout 0.2, and all seeds 42.
The root-level notebook uses 0.002 instead; the src notebook agrees with the
provided comparison table and is the explicit reference for this configuration.

The trainer now supports both the historical Adam/no-scheduler mode and
AdamW/OneCycle. The scheduler is sized to the 282 training batches and steps
after every optimizer update, including the final 8-example batch. The same
OneCycle constructor defaults as Nguyen's code are used (including momentum
cycling). CSVs record current LR, configured max LR, weight decay and scheduler.

MNIST remains native 1-channel 28x28 with ten outputs as specified by the
assignment. Only the first convolution's input width and classifier output
width differ from the CIFAR model. Repeating grayscale into three channels
would not add color information and is not silently done. A literal RGB/100-head
experiment would need its own explicit protocol and result label.

This configuration aligns optimizer settings, not the entire CIFAR pipeline:
training augmentation is still absent, normalization is identity, input/head
sizes differ, and execution is CPU. CIFAR's three-channel mean/std cannot be
copied to a single-channel model. Horizontal flipping may change digit meaning;
matching or adapting that augmentation needs to be documented before claiming
full training-protocol parity. Do not call this a fully controlled comparison.

A completed result is recorded below. To reproduce this optimizer-only
comparison in a fresh output directory:

```powershell
python -m src.train_online --config configs/mnist_nguyen_optimizer.yaml --output data/processed/mnist_nguyen_optimizer_seed42
```

This will reuse the same 9,000/1,000 membership/order as the prior MNIST run,
with no new split or modification of existing saved membership/results.

## Actual validation

```powershell
$env:PYTEST_DEBUG_TEMPROOT=(Resolve-Path .pytest_cache).Path
$env:HF_HUB_OFFLINE='1'
$env:HF_DATASETS_OFFLINE='1'
python -m pytest -v --tb=short
```

Result: **32 passed in 98.77s**. Added checks compare all 282 LR/beta1 values
with Nguyen's direct AdamW/OneCycle constructor, verify scheduler/update counts
including the final batch, and preserve historical Adam defaults.

## Completed optimizer-alignment run

Run: `mnist_nguyen_optimizer_seed42`. Completed all 9,000 training images in
282 updates. Final validation metrics:

- Accuracy: 98.80%.
- Macro precision: 98.7985%.
- Macro recall: 98.8238%.
- Macro F1: 0.988064.
- Training compute: 184.04s; timed run wall: 260.97s.

Saved membership and ordered training rows exactly equal the original 9,000/1,000
MNIST run. The original Adam run scored 96.4%; the new run scores 98.8%, a 2.4
percentage-point increase on the same validation set. Optimizer, schedule and
weight decay changed together; no individual causal effect or multi-seed ranking
is claimed. Preprocessing remains ToTensor only, with identity normalization;
Nguyen's CIFAR crop/flip and normalization are NOT reproduced. Hardware is CPU.
The old comparison table describes the historical Adam run and is no longer
an accurate description of this optimizer-only variant.

Archived source evidence: `results/mnist_nguyen_optimizer_seed42/`.
Notebook: `notebooks/mnist_optimizer_comparison.ipynb`.

Independent checkpoint audit: 988 correct, 12 mistakes on the saved 1,000
validation images. All four metrics match the final CSV within 1e-12, model
state stayed unchanged, memberships match, and official test data was untouched.
Both optimizer comparison notebook code cells executed and passed validation.
