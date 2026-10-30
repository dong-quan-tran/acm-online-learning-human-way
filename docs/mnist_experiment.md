# SmallResNet on MNIST

## Setup and run

From the repository root, use Python with the packages in requirements.txt:

```powershell
python -m pip install -r requirements.txt
python -m pytest -v
python -m src.train_online --config configs/mnist.yaml --output data/processed/mnist_seed42 --download
```

The local project interpreter is `.\.venv\Scripts\python.exe`. The download
flag is only needed when MNIST is absent. Use a new output directory for each
run; existing directories are refused to prevent overwriting results.

## Method

All ten digits are mixed. The official 60000 training images are divided into
54000 training and 6000 validation rows using split seed 42. The official test
set is kept separate and is not evaluated during development. Stream seed 42
shuffles only the training membership. Each training row is used exactly once,
in 1688 batches: 1687 batches of 32 and a final batch of 16. No replay, epochs
beyond the one pass, pretrained weights, or augmentation are used.

MNIST stays at its native 28x28 grayscale resolution. ToTensor scales pixels
to [0, 1]; normalization in the model is identity (mean 0, standard deviation 1).
The SmallResNet residual architecture is extracted from the existing CIFAR
notebook, adapting its first convolution to one channel and its output layer
to ten classes. CIFAR normalization is not applied to MNIST.

The initial standalone baseline uses Adam at learning rate 0.001, dropout 0.2,
cross-entropy, CPU, two PyTorch threads, and deterministic algorithms. All
settings come from configs/mnist.yaml and are copied into each run record.
Validation runs before training, at the first complete batch reaching each
1000-example checkpoint, and at the end. CSV example counts are the actual
counts (for example 1024), rather than rounded checkpoint targets.

Accuracy is the fraction correct. Precision, recall and F1 are macro averages
over all ten classes, assigning zero to undefined class ratios. Evaluation
uses model.eval() and no gradients, and restores the previous training mode.
training_seconds counts model forward/backward/optimizer work, excluding data
loading and validation; total_wall_seconds includes those during the run.

## Outputs

Each isolated run directory contains:

- results.csv: shared result columns plus macro precision, recall and F1.
- run.json: settings, package versions, hardware, command, timing and counts.
- membership.json: exact source row membership and ordered training rows.
- model.pt: final model weights, kept local rather than uploaded.
- online_learning_curve.png: validation accuracy versus examples processed,
  produced with the existing repository plotter.

Outputs live under ignored data/processed so images/checkpoints are not
accidentally included in source commits. Keep the run directory to reproduce
or audit the experiment. The loader can regenerate the order from recorded
seeds; membership.json records it explicitly as well.

## Comparison limits

Nguyen's current shared online trainer and original CIFAR run records are not
published in the available repository branches. This runner supplies reusable
one-pass and evaluation functions, but is not claimed to reproduce that unseen
pipeline. The MNIST optimizer, split and preprocessing above are explicit
provisional choices. Match the published CIFAR settings before claiming a fair
comparison. The older class-incremental notebook is not a comparable one-pass
CIFAR run. The assignment's MNIST-versus-CIFAR results discussion remains
pending those artifacts; a single MNIST seed is an initial run, not a repeated
final estimate.


## Completed results and comparison notebook

Open `notebooks/mnist_cifar_comparison.ipynb` and select Run All to reproduce
saved-result analysis without training. Optional notebook dependencies:
`python -m pip install nbformat nbclient ipykernel`.

The portable submission artifacts are in `results/mnist_seed42_complete/`.
They contain the actual completed CSV, run metadata, original membership, and
learning curve. Weights and downloaded images remain excluded. The archived
run command uses portable paths; the original metadata remains in the local
run folder.

Initial notebook verification executed all four original code cells. The later
saved-checkpoint audit below reran the tests and expanded notebook; no retraining
was performed.
The same-model CIFAR comparison remains pending its unpublished source artifacts;
GaussianNB is displayed only as explicitly labelled context.


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
