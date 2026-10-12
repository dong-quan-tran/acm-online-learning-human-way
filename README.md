# Online Learning – The Human Way

A semester-long ACM Research project studying how an image classifier learns from incoming labeled data. Our main setting is a randomly shuffled, single-pass CIFAR-100 stream with a fixed class set.

## Direction Update — October 12, 2026

We are moving back to online learning following Dr. Athitsos's guidance. Instead of learning separate class groups one after another, we will mix the selected classes throughout the stream.

The team's implementation plan is below. The detailed protocol is awaiting Dr. Athitsos's confirmation; it is not a claim that he has approved every experiment choice.

See [the current direction and protocol](docs/online_learning_direction.md). This README and that document supersede conflicting task-based instructions in older assignment, protocol, and checklist documents. Older documents remain as historical references.

## Research Question

How well can a model learn through small, ongoing updates as labeled data arrives, without repeatedly training on the entire dataset?

## Current Objective

1. Reuse CIFAR-100 and start with classes 0–19 for debugging.
2. Combine those classes before shuffling; do not train Task 1 and Task 2 separately.
3. Feed small batches to one CNN, with one update per batch and one pass through the stream.
4. Do not replay old training examples or use future batches.
5. Measure held-out accuracy versus examples processed.
6. Expand to all 100 classes after the smaller pipeline works.

Catastrophic forgetting is no longer the main metric. EWC, Fisher estimation, and class-blocked experiments are optional later extensions, not current required deliverables. Existing work will be preserved.

## Upcoming Team Tasks

| Member | Next focus |
|---|---|
| Tuong | Explain the online setting, review the protocol, and clarify online versus offline comparisons |
| Khoi | Build the mixed-class stream, seeded ordering, and loader tests |
| Nguyen | Build the shared CNN and single-pass training/evaluation loop |
| Alka | Align the result schema, plot accuracy versus examples processed, and update experiment documentation |
| Dong Quan Tran | Confirm the protocol with Dr. Athitsos, coordinate integration, and track progress |

Progress reviews prioritize effort, commitment, follow-through, communication, collaboration, and response to feedback. Research accuracy and GitHub activity alone do not determine ratings.

## Project Structure

```text
configs/       Recorded experiment settings
data/          Metadata only; do not commit downloaded images
docs/          Current direction, research notes, decisions, and updates
results/       CSV results and generated figures
src/           Data, model, training, evaluation, and plotting code
tests/         Automated checks
```

## Collaboration Rules

- Create a branch from main and open a pull request for changes.
- Include evidence: test output, a result file, a plot, or a clear explanation.
- Do not commit dataset images, checkpoints, secrets, or virtual environments.
- Report completed work, evidence, blockers, and next steps every 2–3 days.
- Check the current direction document before following older assignments.

## Current Completion Checklist

- [ ] Detailed protocol confirmed with Dr. Athitsos.
- [ ] Mixed-class stream works on the initial 20-class subset.
- [ ] Stream tests verify labels, coverage, no repeats, and seeded ordering.
- [ ] CNN completes a single pass with no replay.
- [ ] Evaluation uses held-out data without training on it.
- [ ] Accuracy-versus-examples CSV and plot are saved.
- [ ] Setup and run commands are documented.
- [ ] Full 100-class experiment runs after the debugging stage.
- [ ] Online/offline comparison and repeated-seed results are documented.

These are upcoming goals, not claims that the implementation or tests already pass.
