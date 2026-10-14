# Project Decisions

Updated: October 13, 2026

| Decision | Reason |
|---|---|
| Focus on online learning | Follow Dr. Athitsos's suggestion to study learning from incoming data |
| Keep CIFAR-100 | Reuse the team's existing work |
| Mix classes in random order | Keep the class distribution fixed instead of introducing separate class groups |
| Start with classes 0–19, then use all 100 | Check the small pipeline before expanding |
| Use one pass with no replay | Start with a simple online baseline |
| Track accuracy as examples arrive | Measure learning progress rather than Task 1 forgetting |
| Compare with offline training later | Show what extra passes through the data provide |
| Set EWC aside for now | Keep forgetting prevention as possible future work |
| Start with seed 42 | Make early runs easier to repeat |

## Advisor Confirmation — October 13, 2026

Dr. Athitsos replied:

> Yes, this is exactly what I had in mind.

He confirmed our understanding: mix examples from a fixed class set in random order, learn from small incoming batches in one pass without replay, and track accuracy as more data arrives. EWC remains a possible later extension.

The online-learning direction is confirmed. Batch size, model details, optimizer, evaluation intervals, and comparison budgets remain team implementation choices. Earlier pending-confirmation notes about the direction are superseded by this update.

See `experiment_protocol.md` for experiment rules and
`two_week_assignment.md` for team tasks.
