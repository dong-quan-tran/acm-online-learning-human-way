# Experiment Protocol

Updated: October 13, 2026

## Main Question

How well can a model learn from incoming labeled data without repeatedly
training on the whole dataset?

## Settings

| Item | Plan |
|---|---|
| Dataset | CIFAR-100 |
| Starting subset | Classes 0–19, mixed together |
| Full experiment | All 100 classes after the small run works |
| Model | One small CNN with 100 outputs and original class labels |
| Starting weights | Random initialization |
| Data order | Combine selected classes, then shuffle once |
| Batch size | 32, including the final smaller batch |
| Training | One update per batch and one pass through the stream |
| Loss | Cross-entropy |
| Replay | None |
| Initial seed | 42 |

There are no Task 1 and Task 2 training stages. The same class set remains
available throughout the stream, although class counts vary between batches.

The model learns only from the current batch. It must not train on future
batches or reuse earlier training examples. Model and optimizer state are
kept between updates.

Dr. Athitsos confirmed the online-learning direction on October 13, 2026:
"Yes, this is exactly what I had in mind." Batch size, optimizer, model details,
and evaluation intervals remain team implementation choices.

## Evaluation

- Set aside validation data before creating the training stream.
- Keep training, validation, and test examples separate.
- Check validation accuracy about every 1,000 training examples and at the end.
- Evaluation must not change weights or running statistics.
- Use the official test set for the final experiment, not for tuning.
- Repeat final runs with three recorded seeds.

Main plot: accuracy versus training examples processed.

Also record final accuracy, update count, training time, and memory use.
State how memory was measured.

## Results

Save one row per evaluation checkpoint:

```csv
mode,run_id,model_seed,stream_seed,num_classes,examples_processed,updates,passes,batch_size,learning_rate,optimizer,eval_split,accuracy,training_seconds
```

Accuracy is a decimal from 0 to 1. Save each run's configuration with its results.

## Offline Comparison

After the online run works, compare it with multiple-pass offline training.

Use the same model, data splits, and image transforms. Report the extra
passes, updates, and time. This is not an equal-training-budget comparison.

As a code check, one-pass offline training should closely match online
training when initialization, batches, and updates are identical.

## Checks and Run Records

- Verify labels, image shapes, stream order, and sample counts.
- Check that each training example appears once and splits do not overlap.
- Check that evaluation leaves the model unchanged.
- Make sure the trainer and plotter use the same result format.
- Record seeds, settings, package versions, device, and exact commands.
- Agree on the validation split, transforms, optimizer, learning rate,
  and offline pass count before comparing results.
- Do not upload dataset images, checkpoints, secrets, or virtual environments.

EWC, Fisher estimation, and forgetting metrics are not required in this phase.
