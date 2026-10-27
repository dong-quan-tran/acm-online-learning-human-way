## Goal

Use the current online-learning pipeline to answer:

1. How does SmallResNet perform on MNIST?
2. How do two non-neural models compare with SmallResNet on CIFAR-100?

## Shared Rules

- Train in one pass, batch size 32, with no replay.
- Mix the selected classes before shuffling.
- Start with seed 42 and record all settings.
- Keep training, validation, and test data separate.
- Use validation data for development—not the test set.
- Save results in the shared CSV format.
- Start recording results in terms of Accuracy, F1 score, Precision, Recall.
- Record training time, device, package versions, and exact commands.
- Keep each experiment’s results separate.

The goal is a **fair comparison**, not a required accuracy score.

## Tuong — Review 2–3 Papers

### Tasks
- Review two papers; a third is optional.
- Choose papers about single-pass online learning, learning from limited data, or simple online classifiers.
- Confirm the paper list with Johnny before writing.
- Check whether each paper uses replay, pretraining, or repeated training passes.

### For each paper, explain
- What problem.
- What model and dataset.
- How its online-learning rules differ from ours.
- Its main finding.
- One idea we could test.
- Any limits that prevent a direct comparison.

### Submit
- One report per paper (no format, put anything on there, as long as you can explain it).
- A short comparison table.
- Two practical recommendations, stating what we would change and why.

Done when: the team understands which ideas fit our setting and which do not.

## Khoi — Run SmallResNet on MNIST

### Tasks
- Reuse the shared online-training and evaluation code.
- Load all ten MNIST digit classes and shuffle them together.
- Adapt the model to one input channel and ten outputs.
- Replace CIFAR-specific preprocessing with documented MNIST preprocessing.
- Train from scratch with one pass and no replay.
- Keep batch size 32 and the same evaluation process.
- Record any other changes from the CIFAR setup.
- Save the MNIST run separately from the existing CIFAR run.

### Submit
- MNIST loader support and configuration.
- Tests for labels, image shape, split separation, and sample coverage.
- Results CSV and accuracy-over-time plot.
- Exact setup and run commands.
- A short summary comparing the two dataset experiments.

Done when: another member can run the MNIST experiment without editing code. Focus on what the difference is.

## Nguyen — Run Online Logistic Regression on CIFAR-100

### Tasks
- Use `SGDClassifier(loss="log_loss")`.
- Use the finalized CIFAR classes 0–19 split and recorded stream order.
- Flatten images into vectors and scale pixel values consistently.
- Update the same model with one `partial_fit()` call per batch.
- Supply the full class list on the first call.
- Do not use full-dataset `fit()`, restart between batches, or replay examples.
- Evaluate at the same example counts as the shared CIFAR baseline.
- Start with raw pixels; do not add pretrained features.

### Submit
- Training code and configuration.
- Tests for input shape, class handling, and updates across batches.
- Results CSV and learning curve.
- Training time, device, and exact commands.
- A short comparison with SmallResNet.

Done when: the model completes the shared stream once and produces results the existing plotter can read.

## Alka — Run Online Gaussian Naive Bayes on CIFAR-100

### Tasks
- Use `GaussianNB`.
- Use the same CIFAR split, stream order, and batch size as Khoi.
- Use the same flattened images and pixel scaling as logistic regression.
- Update the same model once per batch with `partial_fit()`.
- Supply the full class list on the first call.
- Evaluate at the shared checkpoints.
- Check for warnings or invalid predictions.
- Reuse the existing CSV format and plotting tools.

### Submit
- Training code and configuration.
- Basic update and prediction tests.
- Results CSV and learning curve.
- Training time, device, and exact commands.
- A short comparison with logistic regression and SmallResNet.

Done when: the model finishes one pass and its results can be plotted without manual edits.

For both non-neural models, mark pre-training accuracy as unavailable if prediction requires an initial update. Do not label an already-trained result as “zero examples.”

### Final summaries

| Summary | Include |
|---|---|
| Dataset comparison | SmallResNet on CIFAR versus MNIST, with dataset and preprocessing differences stated |
| Model comparison | SmallResNet, logistic regression, and GaussianNB on the same CIFAR stream |

Report accuracy, training examples, training time, and hardware. Explain any differences in preprocessing or compute.
