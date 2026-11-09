# Project Decisions

Updated: November 9, 2026

## Current Decision: Two Separate TTT Experiments

Dong Quan reports that Nguyen and Tuong presented their plans to Dr. Athitsos and received approval to proceed. The exact meeting date is not recorded here. This is a project-lead report of approval, not a quoted advisor statement.

| Decision | Meaning |
|---|---|
| Begin rotation-based test-time training | Learn from rotation targets without using test class labels |
| Keep two experiments separate | Nguyen uses CIFAR-10/CIFAR-10-C; Tuong uses CIFAR-100 classes 0-19 with clean and fixed-noise conditions |
| Use matched frozen baselines | Separate the effect of joint source training from test-time updates |
| Check evaluation before full runs | Keep labels outside adaptation and state prediction timing and reset rules |
| Treat implementation settings as open until recorded | Dataset splits, budgets, learning rates, steps, and noise settings are team choices |
| Preserve prior work | Do not overwrite historical results or delete member branches during closeout |

The earlier November 1-7 assignment asked Nguyen for a small Tent feasibility trial. The newly approved TTT direction replaces that trial as his next experiment. Credit the presentation and approved plan separately from implementation; the original trial must not be marked completed without evidence. Tent/EATA remain possible later comparisons.

Tuong's and Khoi's submitted reviews cover one paper each. The original two-paper requirement remains a recorded gap, not a completed item. The coordinator should state whether to carry it forward or waive it; this update does not decide that.

Current work: [November 9 assignment](11_9_assigment.md).
Closeout: [November 1-7 wrap-up](weekly_updates/2026-11-07_wrap_up.md).

## Historical Decision: October 13, 2026

Dr. Athitsos replied about the earlier labeled online-learning setting:

> Yes, this is exactly what I had in mind.

That setting mixes examples from a fixed class set in random order, learns from small labeled batches in one pass without replay, and tracks accuracy as examples arrive.

Historical choices were CIFAR-100, classes 0-19 for debugging, a later expansion to all 100 classes, an initial seed of 42, and a later offline comparison. EWC was deferred. Model details, batch size, optimizer, evaluation intervals, and budgets remained team choices.

The older experiment protocol and two-week assignment are historical references. They must not be read as the current TTT protocol.
