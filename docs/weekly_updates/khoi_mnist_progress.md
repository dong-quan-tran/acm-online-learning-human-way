# Completed MNIST run — current status

The `mnist_seed42_complete` experiment has finished: 54,000 training examples,
1,688 updates, final validation accuracy 99.0833%, macro precision 99.1030%,
macro recall 99.0723%, and macro F1 99.0846%. Training compute was 2895.75s;
run wall time was 5430.79s on CPU. All five local run artifacts exist.
Saved membership was checked: 54,000 unique training rows, 6,000 unique
validation rows, zero overlap, covering all 60,000 source training rows.

The user's terminal evidence reports 23 passed, 2 warnings in 319.83s.
The warnings themselves were not provided. This is not a fresh test-suite rerun.

See `notebooks/mnist_cifar_comparison.ipynb` for executed saved-result analysis
and `results/mnist_seed42_complete/` for the CSV, run metadata, exact membership
and original chart. Model weights and dataset images remain local and ignored.
The notebook's GaussianNB comparison is context only; the requested same-model
CIFAR comparison and shared trainer parity still await Nguyen's online artifacts.

The sections below are historical progress notes and do not describe the final
run's status.

---

# Khoi MNIST assignment progress

**Update:** the standalone one-pass runner is now implemented and the initial
real MNIST run has started. See `docs/mnist_experiment.md` for its method and
commands. The earlier pending-work list below records the starting state.
Shared CIFAR pipeline parity and the dataset comparison remain pending.

Started 2026-10-27 on `khoi/online-data-loader`, from a clean working tree.

## Implemented

- `src/mnist_data.py`: all ten digits, native 28x28 grayscale images,
  `ToTensor()` scaling to [0, 1], `(image, label, source-row ID)` batches.
- Fixed seeded 54000/6000 membership from the official 60000 training images.
  This is a documented MNIST starting choice, independent of the pending CIFAR
  membership. Split seed and stream seed are separate. The official 10000 test
  images remain separate. No second DataLoader shuffle; final batches retained.
- `src/models.py`: extracted SmallResNet's convolution/residual/pooling layout
  from the existing class-incremental notebook, with one input channel and ten
  outputs. No pretraining. No CIFAR normalization on MNIST: model normalization
  defaults to identity, matching the loader's [0, 1] pixels. The old notebook
  remains unchanged. This is not claimed to be Nguyen's unpublished online model.
- `configs/mnist.yaml`: dataset/model/split settings; one pass, no replay,
  batch size 32, seeds 42. Optimizer and evaluation settings remain pending.
- `tests/test_mnist.py`: offline fixture-based coverage/order/shape/label/ID
  checks and model training-update / evaluation-state checks.

## Verified locally

MNIST was downloaded through torchvision into ignored `data/raw/MNIST`.
A separate real-data smoke check loaded each split and ran one training update
on an actual MNIST batch (not a completed experiment):

```text
train samples 54000 batches 1688 shape (32, 1, 28, 28)
validation samples 6000 batches 188 shape (32, 1, 28, 28)
test samples 10000 batches 313 shape (32, 1, 28, 28)
overlap 0 coverage 60000
training labels [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
training example IDs: train:12827, train:5321
validation example IDs: train:37542, train:43087
test example IDs: test:0, test:1
one-update logits (32, 10), finite loss True
CUDA available: False
```

Train has 1687 full batches plus a final batch of 16. No model predictions
were evaluated on the official test set; the smoke check only loaded its batch.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_mnist.py -v
# 6 passed in 11.03s
```

Quick loader usage (download is explicit):

```python
from src.mnist_data import get_mnist_loader
from src.models import SmallResNet

train = get_mnist_loader(download=True)
validation = get_mnist_loader("validation", download=True)
model = SmallResNet(in_channels=1, num_classes=10)
images, labels, sample_ids = next(iter(train))
```

## Remaining assignment work

1. Obtain the current shared online trainer/evaluator and CIFAR run settings.
   Refetched origin and inspected `origin/nguyen/small-resnet` at `8060729`:
   the current online pipeline and run records are still absent.
2. Connect this loader/model/config to that trainer. Match optimizer, learning
   rate, evaluation checkpoints and metric definitions, recording MNIST changes.
   Save exact MNIST split membership and stream order with the run configuration.
3. Run the complete 54000-example stream once, from scratch, with no replay.
   Record validation accuracy, macro precision/recall/F1, training time, device,
   versions and exact commands. No completed-run accuracy is available yet.
4. Save a separate MNIST CSV in the shared format and a learning curve.
5. Compare against the published CIFAR results, stating dataset, preprocessing,
   sample-count and compute differences. Provide one runnable training command.

The assignment is started, not complete. No completed experiment or CIFAR
comparison is claimed by the loader/model smoke checks.

Full regression command and actual result:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:HF_DATASETS_OFFLINE='1'
$env:PYTEST_DEBUG_TEMPROOT=(Resolve-Path .pytest_cache).Path
.\.venv\Scripts\python.exe -m pytest -v
# 20 passed in 33.52s
```

## Initial experiment started

Implemented src/train_online.py with reusable one-pass training, validation,
macro metrics and isolated artifacts. Training/evaluation settings are explicit
provisional MNIST choices, not verified against the unpublished CIFAR trainer.

```powershell
.\.venv\Scripts\python.exe -u -m src.train_online --config configs/mnist.yaml --output data/processed/mnist_seed42_initial
```

The full real-data run started on CPU. Its initial validation accuracy was
0.0956667; at 4000 training examples it was 0.9111667. These are intermediate
checkpoints, not final results. Follow results.csv in that run directory for
actual saved checkpoints. Completion should produce model.pt and the learning
curve; their existence must be checked before claiming the run finished.

The extended full regression suite (same offline/cache environment above):

```powershell
.\.venv\Scripts\python.exe -m pytest -v
# 23 passed in 32.67s
```

New training checks cover known confusion-matrix metrics, unchanged model
state during evaluation, one-pass update counts, and final-checkpoint behavior.

## Separate progress branch

MNIST source, configuration, documentation and tests are being saved on
`codex/mnist-assignment`, branched from Khoi's completed loader work at
`838e284`. At this snapshot the real run is still in progress (latest inspected
checkpoint: 33024 of 54000 examples, validation accuracy 0.9751667).
No final training result is claimed. Dataset files, live CSV, saved row order
and model artifacts remain local under ignored data/raw and data/processed.
The source commit includes the reproducible run command, not a completed run.


## Saved-checkpoint reliability audit

Re-evaluated the original `model.pt` on exactly the saved 6,000 validation
source rows, loaded directly from the original MNIST training source. Both
configured loader memberships exactly match the saved rows; zero overlap.
The checkpoint achieved 5,945 correct and 55 mistakes. All four metrics
reproduced the final CSV within 1e-12; model state remained unchanged.
No retraining or official test-set evaluation was performed.

```powershell
python -m src.audit_mnist --run data/processed/mnist_seed42_complete --output results/mnist_seed42_complete/audit
$env:PYTEST_DEBUG_TEMPROOT=(Resolve-Path .pytest_cache).Path
$env:HF_HUB_OFFLINE='1'
$env:HF_DATASETS_OFFLINE='1'
python -m pytest -v --tb=short
```

Fresh suite: **23 passed in 48.87s**, no warnings reported. The earlier two
warning messages were not available, so no cause is inferred.
`audit/` contains exact predictions, the checkpoint SHA-256, matching metrics,
a confusion matrix and first 20 mistakes. The comparison notebook was rerun:
all five code cells executed and passed notebook validation. Diagnostic plots
were visually checked. This confirms this run's consistency, not multi-seed
robustness or a controlled MNIST/CIFAR comparison.
