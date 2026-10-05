# Online Learning – The Human Way

A semester-long ACM Research project studying catastrophic forgetting in continual learning. We compare normal sequential training against Elastic Weight Consolidation (EWC) on Split CIFAR-100.

## Research Question

Can standard EWC reduce catastrophic forgetting when a model learns CIFAR-100 tasks one after another?

## Current Two-Week Objective

Build a reproducible two-task experiment:

1. Load CIFAR-100 and split it into sequential tasks.
2. Train a baseline CNN on Task 1, classes 0–9, then Task 2, classes 10–19.
3. Measure Task 1 forgetting after Task 2 training.
4. Implement standard EWC and compare it fairly against the baseline.

## Team Assignments

| Member | Week 1 | Week 2 |
|---|---|---|
| Tuong | EWC and EWC-DR guide | Fisher-information guide and EWC code review |
| Khoi | CIFAR-100 loader, task split, and data tests | Fisher loader, data safeguards, and reproducibility checks |
| Nguyen | Small CNN and sequential baseline | Standard EWC training script and lambda experiments |
| Alka | Protocol, result schema, and baseline plot | Comparison plots, results summary, and documentation review |
| Dong Quan Tran | Project integration, task tracking, code review, and final deliverables | Project integration, task tracking, code review, and final deliverables |

## Project Structure

```text
configs/       Experiment settings
data/          Task-split metadata only; do not commit CIFAR-100 images
docs/          Protocol, research notes, decisions, and weekly updates
results/       CSV results and generated figures
src/           Data, model, training, evaluation, and plotting code
tests/         Automated checks
```

## Branch and Pull Request Rules

- Create a branch from `main` before starting work.
- Use branch names such as `tuong/ewc-notes`, `khoi/cifar100-data`, `nguyen/baseline-cnn`, and `alka/protocol-plots`.
- Open a pull request to `main` when work is ready.
- Include evidence in the pull request: test output, a plot, a result file, or a short explanation.
- Do not commit CIFAR-100 images, checkpoints, API keys, or virtual-environment folders.

## Progress Update Format

Post an update in the group every 2–3 days:

```text
Name:
Week:
Completed:
Evidence: GitHub PR, commit, file path, screenshot, plot, or output
Blocker:
Next step:
```

## Two-Week Completion Checklist

- [ ] CIFAR-100 downloads automatically.
- [ ] Task 1 and Task 2 splits are verified.
- [ ] Data tests pass.
- [ ] Baseline CNN completes Task 1 → Task 2.
- [ ] Baseline results and forgetting plot are saved.
- [ ] Fisher values are calculated and checked.
- [ ] Standard EWC completes Task 1 → Task 2.
- [ ] Multiple EWC lambda values are tested.
- [ ] Baseline and EWC plots are saved.
- [ ] README and experiment documentation are complete.