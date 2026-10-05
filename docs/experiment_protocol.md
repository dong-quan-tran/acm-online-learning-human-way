# Experiment Protocol

## Research Question

Can standard Elastic Weight Consolidation reduce catastrophic forgetting when a model learns CIFAR-100 tasks one after another?

## Initial Experiment

- Dataset: CIFAR-100
- Task 1: class IDs 0–9
- Task 2: class IDs 10–19
- Model: one shared small CNN with 100 output classes
- Learning order: train Task 1, evaluate Task 1, train Task 2 using the same model, then evaluate Tasks 1 and 2
- Baseline: cross-entropy loss only; no replay and no EWC
- EWC: cross-entropy plus an EWC penalty during Task 2
- Initial random seed: 42

## Required Measurements

- Task 1 accuracy after Task 1 training
- Task 1 accuracy after Task 2 training
- Task 2 accuracy after Task 2 training
- Task 1 forgetting = Task 1 accuracy after Task 1 minus Task 1 accuracy after Task 2
- Runtime and configuration values, if available

## Fair Comparison Rules

The baseline and EWC runs must use the same model, dataset split, transforms, seed, optimizer, epochs, batch size, and learning rate. Only the EWC penalty and lambda value may differ.

## Reproducibility Rules

Record the Python version, package versions, device, seed, configuration values, and exact run command. Do not upload CIFAR-100 files, checkpoints, secrets, or `.venv` folders.
