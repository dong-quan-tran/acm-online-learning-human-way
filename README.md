# Online Learning - The Human Way

An ACM Research project studying how image classifiers learn from incoming data.

## Current Phase: Test-Time Training

Updated: November 9, 2026

The project is moving from labeled, single-pass learning to adaptation using unlabeled test images. Dong Quan reports that Nguyen and Tuong presented their plans to Dr. Athitsos and received approval to begin experiments. This records approval of the direction, not approval of every implementation setting.

Keep the two experiments separate:

| Experiment | Lead | Scope |
|---|---|---|
| A | Nguyen | Train on CIFAR-10 and evaluate rotation-based TTT on CIFAR-10-C, with a clean control |
| B | Tuong | Use CIFAR-100 classes 0-19; compare frozen models, standard TTT, and online TTT on clean images and fixed Gaussian noise |

Khoi owns data and evaluation checks. Alka owns result logging, plots, and comparisons. These are the roles in the current assignment.

- [Current team assignment](docs/11_9_assigment.md)
- [November 1-7 wrap-up](docs/weekly_updates/2026-11-07_wrap_up.md)
- [Decision log](docs/decisions.md)

The assignment filename is kept unchanged to preserve existing links. Source-training budgets, corruption settings, adaptation settings, and final evaluation rules must be agreed before full runs.

## Main Question

Can a trained classifier improve on changed images by learning from those images without using their class labels?

Compare adaptation against the same jointly trained model with no test updates. Keep class labels outside adaptation and use them only for scoring. Report accuracy, cases where adaptation hurts, update counts, and runtime. A lower rotation loss alone is not proof of better classification.

## Current Status

- Literature reviews and presentation plans are available across submissions and member branches.
- Advisor approval to begin the two experiments is reported by the project lead.
- Report collection and integration are still pending.
- This documentation update does not certify implemented TTT, completed adaptation runs, or passing tests.

## Earlier Phase: Labeled Online Learning

On October 13, 2026, Dr. Athitsos confirmed the earlier direction: mixed classes, small incoming labeled batches, one pass, and no replay. Existing code and results remain useful historical baselines.

The older [direction](docs/online_learning_direction.md), [protocol](docs/experiment_protocol.md), and [assignments](docs/two_week_assignment.md) describe that phase. They do not override the new TTT assignment. Full 100-class expansion, offline comparisons, and EWC are not the immediate TTT milestone. Preserve earlier results and label new runs separately.

## Project Structure

```text
configs/       Experiment settings
data/          Metadata only; no downloaded images
docs/          Assignments, decisions, notes, and weekly updates
results/       CSV results and generated figures
src/           Data, model, training, evaluation, and plotting code
tests/         Automated checks
```

## Collaboration Rules

- Work on a branch and submit a pull request.
- Include exact commands, settings, and evidence.
- Separate submitted, reviewed, merged, and reproduced work.
- Do not commit dataset images, model weights, secrets, or virtual environments.
- Preserve original runs; use a new run ID for changed settings.
- Report progress and blockers every 2-3 days.
- Credit effort, understanding, communication, and follow-through, not accuracy alone.

## Next Milestone

Each experiment should complete one small, reproducible frozen-versus-adapted pilot. Check label separation, model resets, sample coverage, and result logging before expanding to three seeds.
