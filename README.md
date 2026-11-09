# Online Learning - The Human Way

ACM Research project on learning from incoming images.

## Current Work

Nguyen and Tuong presented to Dr. Athitsos and received approval to begin two separate test-time training experiments, as confirmed by Dong Quan.

| Lead | Experiment |
|---|---|
| Nguyen | CIFAR-10 training and rotation-based adaptation on CIFAR-10-C |
| Tuong | CIFAR-100 classes 0-19: frozen models, standard TTT, and online TTT on clean and noisy images |

Khoi handles data and evaluation checks. Alka handles logging, plots, and comparisons.

- [Current assignment](docs/11_9_assigment.md)
- [Weekly wrap-up and remaining work](docs/weekly_updates/2026-11-07_wrap_up.md)
- [Decisions](docs/decisions.md)
- [Received report summaries](docs/reports/2026-11-07/README.md)

## Rules

- Keep the two experiments and historical runs separate.
- Keep test class labels outside adaptation.
- Compare adaptation against the same jointly trained frozen model.
- Save settings, seeds, results, and exact run commands.
- Report failures and accuracy drops as well as improvements.
- Do not commit datasets, model weights, secrets, or environments.
- Preserve active member branches and original work.

Earlier labeled online-learning protocols and EWC notes are historical references, not the current assignment. First complete one checked pilot per experiment, then repeat across seeds.
