# Literal Nguyen CIFAR protocol applied to MNIST

This is a separate experiment. No original configs, checkpoints or results are
replaced. Reference: origin/main 3d356f9, `src/rawsgd-cifar-100.ipynb`.
The root notebook uses a different max LR (0.002); the src version matches the
provided table (0.0005), and is the reference used here.

## Matched settings

- Three-channel 32x32 model input; identical 64/128/256 residual architecture,
  fixed 100-output head; dropout 0.2. Predictions use all 100 outputs, no masking.
- CIFAR mean [0.5071,0.4865,0.4409], std [0.2673,0.2564,0.2762] inside the model.
  Loader scaling only; no second normalization.
- Train: RandomCrop(32, padding=4, reflect), RandomHorizontalFlip(default p=0.5),
  float32 [0,1] scaling. Validation: scaling only, no random augmentation.
- AdamW, max LR 0.0005, weight decay 0.0005; OneCycleLR pct_start=0.15 and the same
  PyTorch constructor defaults, one scheduler step after each of 282 updates.
- Batch size 32, final batch 8, one pass/no replay. Model/split/stream seed 42.
- Validation batch 256, two workers. Fixed stream order and train loader
  generator seed 42; eval generator omitted, matching Nguyen's code.
- Deterministic algorithms, cudnn deterministic, benchmark disabled;
  CUBLAS_WORKSPACE_CONFIG=:4096:8. CUDA if available, otherwise explicit CPU fallback.
- Checkpoints: 0,1024,2016,3008,4000,5024,6016,7008,8000,9000 examples.

## Necessary adaptation and interpretation

MNIST has grayscale 28x28 source images: resize to 32x32 (bilinear, antialias)
and replicate grayscale to three channels BEFORE Nguyen's uint8-tensor transform.
This does not add color information. MNIST still has ten real digit labels,
0-9, although the model has 100 outputs. Incorrect predictions into outputs
10-99 count as errors. Macro precision/recall/F1 average over the ten real
classes; absent outputs are not averaged as 90 zero-score classes.

Use Nguyen's per-class NumPy split procedure with seed 42, adapted to select
1,000 rows per digit: first 100 validation, next 900 training; sort membership
before the torch stream permutation. This gives the same total 9,000/1,000
counts, a 10% holdout and balanced digit classes, but a NEW saved MNIST
membership relative to prior random-split runs. Dataset identities cannot be
shared between MNIST and CIFAR. The remaining 50,000 MNIST training rows are
unused; the official test set is untouched. The difference from CIFAR's
20 classes (450/50 per class) is explicit.

Horizontal flips can make digit images semantically unusual. CIFAR statistics
are not MNIST statistics. These choices intentionally match the requested
literal baseline, not a recommended MNIST-specific augmentation policy. Report
this as a separate protocol variant, not a guaranteed fair dataset comparison.
Even with the matched input/head, task classes, source images, source selection,
hardware and package versions can differ. Timing is not directly comparable
across devices. We do not reproduce the notebook's train-mode dummy shape
forward that can modify BatchNorm before training.

```powershell
python -m pytest -v --tb=short
python -m src.train_online --config configs/mnist_nguyen_literal.yaml --output data/processed/mnist_nguyen_literal_seed42
python -m src.audit_mnist --run data/processed/mnist_nguyen_literal_seed42 --output results/mnist_nguyen_literal_seed42/audit
```

All historical native-MNIST and optimizer-only experiments remain separate.

## Source consistency and hardware

The available published `results/smallresnet-cifar-100.csv` at origin/main
3d356f9 records max LR 0.002 and final accuracy 0.395, run
`online_c20_m42_s42_20261015-200612`. It is NOT the screenshot's 0.0005/0.432
run. The new configuration follows the src notebook and supplied table's
0.0005 max LR. Do not label the available CSV as a verified matched-0.0005
CIFAR result. The corresponding original run record is still needed.

CUDA availability check on this machine: False; the explicit auto-device
fallback is CPU. No CUDA result or hardware-matched timing is claimed.

## Completed literal-profile run

Run: `mnist_nguyen_literal_seed42`. Final accuracy: **96.50%**;
macro precision **96.5328%**, macro recall
**96.5000%**, macro F1 **0.964973**.
Completed 9,000 examples / 282 updates, including the final batch of 8.
CPU training compute: 157.63s; timed run wall:
365.37s. Worker startup, loading and validation contribute
to wall time, not training compute. Neither is a CUDA timing comparison.

Regression suite for the profile implementation: **35 passed in 62.68s**.
The run/CSV/configuration/membership/plot are archived under
`results/mnist_nguyen_literal_seed42/`. Exact selected source rows are retained.
The historical 96.4% and optimizer-only 98.8% native-MNIST results remain separate.
No individual-cause claim is made for accuracy differences because multiple
protocol components and the validation membership changed.

Checkpoint audit reproduced all four metrics to 1e-12: 965 correct, 35 errors,
zero outside-digit predictions, no overlap or model updates. Real source class
counts are exactly 900 train / 100 validation per digit. All three notebook code
cells executed and passed notebook validation. Official test images were unused.
