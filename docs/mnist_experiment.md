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
