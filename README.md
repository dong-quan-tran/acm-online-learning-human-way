# Online Learning - The Human Way

ACM Research project on learning from incoming images.

## Current Work

Nguyen and Tuong presented to Dr. Athitsos and received approval to begin two separate test-time training experiments, as confirmed by Dong Quan.

| Lead | Experiment |
|---|---|
| Nguyen | CIFAR-10 training and rotation-based adaptation on CIFAR-10-C |
| Tuong | CIFAR-100 classes 0-19: frozen models, standard TTT, and online TTT on clean and noisy images |

Khoi handles data and evaluation checks. Alka handles logging, plots, and comparisons. TTT is planned; the existing runners below are labeled-learning baselines, not TTT implementations.

- [Current assignment](docs/11_9_assigment.md)
- [Weekly wrap-up and remaining work](docs/weekly_updates/2026-11-07_wrap_up.md)
- [Decisions](docs/decisions.md)
- [Received report summaries](docs/reports/2026-11-07/README.md)

## Quick Start

From the repo root, use an activated virtual environment:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.train_online --config configs/mnist_9000.yaml --output data/processed/mnist_9000_new_run --download
```

The training command runs the 9,000/1,000 MNIST baseline and saves metrics, settings, membership, weights, and a plot in the output folder. Choose a new output folder for each run; existing folders are refused. Tests may download CIFAR-100, and the training command may download MNIST. These commands were checked against the code but not executed during cleanup. Package versions are not pinned.

To plot the saved CIFAR comparison without training, use a POSIX shell:

```bash
RESULTS_CSV=results/gaussiannb_vs_resnet_results.csv FIGURE_NAME=comparison_check.png python src/plot_results.py
```

This writes results/figures/comparison_check.png. Do not overwrite or commit a generated figure until you have checked it.

## Which Files to Use

- src/train_online.py: MNIST training only; [details](docs/mnist_experiment.md).
- src/train_gaussiannb.py: CIFAR-100 GaussianNB; [commands and settings](docs/gaussiannb_results_summary.md).
- src/rawsgd-cifar-100.ipynb: reference notebook for the updated CIFAR SmallResNet run, not a packaged shared-loader runner.
- rawsgd-cifar-100.ipynb: earlier CIFAR run with different settings; keep its results separate.
- src/naive-fine-tune-class-incremental-cifar-100.ipynb: historical class-incremental experiment.
- notebooks/: saved-result analysis, not new training runs.
- tests/: automated checks; results/: saved run evidence.

## Rules

- Keep the two experiments and historical runs separate.
- Keep test class labels outside adaptation.
- Compare adaptation against the same jointly trained frozen model.
- Save settings, seeds, results, and exact run commands.
- Report failures and accuracy drops as well as improvements.
- Do not commit datasets, model weights, secrets, or environments.
- Preserve active member branches and original work.

Earlier labeled online-learning protocols and EWC notes are historical references, not the current assignment. First complete one checked pilot per experiment, then repeat across seeds.
