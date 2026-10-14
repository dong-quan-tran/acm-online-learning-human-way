# Week 2 Assignment: Move to Online Learning

Updated: October 13, 2026  
Duration: One week

## Goal

Build on Week 1's work and adapt the project to Dr. Athitsos's confirmed
online-learning setting.

Instead of training separate class groups one after another, we will mix
the selected classes and update the model as small batches arrive.

This week's required result is one working 20-class online run, with tests,
a results CSV, and an accuracy-over-time plot.

Keep Week 1 contributions. Reuse and adapt them rather than starting over.

## What Changes

| Previous plan | New plan |
|---|---|
| Train classes 0–9, then classes 10–19 | Mix classes 0–19 before training |
| Separate Task 1 and Task 2 stages | One continuous stream |
| Train repeatedly on each task | One pass through the mixed stream |
| Measure Task 1 forgetting | Measure accuracy as examples are processed |
| Implement EWC and Fisher estimation | Use cross-entropy; defer EWC |

The same class set remains available throughout the stream.
We are no longer using separate class-incremental training stages.

Dr. Athitsos confirmed our understanding on October 13, 2026:
"Yes, this is exactly what I had in mind." The direction is confirmed;
specific implementation settings remain team choices.

## Shared Settings

| Item | Setting |
|---|---|
| Dataset | CIFAR-100, fine labels |
| Required run | Classes 0–19, mixed together |
| Model | Nguyen's SmallResNet with 100 outputs |
| Labels | Original CIFAR-100 class IDs |
| Starting weights | Random initialization |
| Data order | Combine training examples, then shuffle once |
| Batch size | 32, including the final smaller batch |
| Training | One update per batch, one pass |
| Loss | Cross-entropy |
| Replay | None |
| Initial seed | 42 |

Keep the same model and optimizer throughout the stream.
Do not restart between batches, expand the classifier, or replay old batches.

Model and optimizer state may be kept between updates.

Before running experiments, agree on:
- Validation size and split seed.
- Image transforms and normalization.
- Optimizer and learning rate.
- Scheduler, if used.
- Model and stream-order seeds.

Record these settings in `configs/online.yaml`.
Use `experiment_protocol.md` for the shared rules.

## Scope for This Week

Required:
- Adapt the existing data and model code.
- Complete one 20-class single-pass run.
- Verify data separation and evaluation.
- Save results and a real learning curve.
- Document commands another member can run.

Stretch goals:
- Expand to all 100 classes.
- Repeat with more seeds.
- Compare with multiple-pass offline training.

Do not delay the required small run to complete stretch goals.

# Team Tasks

## Tuong — Explain the New Setting and Review the Method

### Build on Week 1

Keep your EWC guide as background. It remains useful work, but EWC is not
this week's required method.

### Tasks

1. Write `docs/online_learning_guide.md` using simple words.
2. Explain:
   - How learning from incoming batches works.
   - Why classes are mixed instead of separated into tasks.
   - What one pass and no replay mean.
   - How this differs from repeatedly training on the whole dataset.
   - Why learning progress replaces forgetting as the main focus.
3. Include this flow:

```text
Receive labeled batch
→ Update the model once
→ Move to the next batch
→ Check held-out accuracy at recorded intervals
```

4. Add a short note to the EWC guide saying it is background for possible
   future work.
5. Review the protocol and Nguyen's training changes.
6. Check that:
   - There are no separate class-based training stages.
   - The classifier is fixed throughout the run.
   - Training makes one pass without replay.
   - Validation and test examples are not used for training.
   - Evaluation does not change the model.
7. Help Alka explain the results and limits clearly.

### Submit

- Short online-learning guide.
- EWC status note.
- Review comments on the protocol or training pull request.
- Any unresolved method questions.

### Done When

The team can understand the new setting, and problems found during review
are clearly recorded.

## Khoi — Adapt the Existing Data Loader

### Build on Week 1

Reuse `src/data.py`, the dataset download, and class filtering.
Keep useful subset functions, but do not use them as separate training stages.

### Tasks

1. Select classes 0–19 and combine their examples before shuffling.
2. Keep original class IDs. Do not remap each group separately.
3. Reserve validation examples from the training split before creating the stream.
4. Record the validation split seed and sample membership.
5. Provide separate training, validation, and test loaders.
6. Make the training order repeatable with a recorded stream seed.
7. Include every selected training example once, including the final smaller batch.
8. Do not oversample classes or require every batch to contain every class.
9. Agree on the loader interface with Nguyen before integrating.
10. Agree on image transforms and normalization:
    - Nguyen's model already normalizes its inputs.
    - Do not apply the same normalization again in the loader.
    - Keep input type and scaling consistent.
11. Print a short setup summary:
    - Selected classes.
    - Examples per split.
    - Batch size and number of training batches.
    - Split and stream seeds.
12. Document the dataset package and cache location.

### Tests

Check:
- Only selected class labels appear.
- Image tensors have the expected shape, type, and scaling.
- Training, validation, and test examples do not overlap.
- Every training example appears exactly once.
- The final smaller batch is included.
- The same seed produces the same order.
- Different seeds produce different orders.

Use sample identities to check coverage and overlap, not labels alone.

### Submit

- Updated data loader.
- Data and split tests.
- Test output.
- Short data setup instructions.

### Done When

Nguyen can use the mixed stream directly, and tests verify labels,
coverage, split separation, and repeatable order.

## Nguyen — Adapt the Week 1 Model and Training Code

### Build on Week 1

Your notebook already includes:
- SmallResNet.
- Image transforms and seed settings.
- Training and evaluation functions.
- Completed Task 1 and Task 2 runs.
- Saved checkpoints and diagnostic plots.

Keep the notebook as the record of the earlier experiment.
Adapt its reusable parts for this week's online run.

### 1. Move Reusable Code into the Repo

- Put SmallResNet and its supporting layers in `src/model.py`.
- Put shared evaluation code in `src/evaluate.py`.
- Create `src/train_online.py`.
- Replace the hard-coded Kaggle path with a project-relative path.
- Bring the submitted code into the shared repo if it is not already there.
- Do not commit downloaded images or checkpoints.

### 2. Use One Fixed Model

- Initialize `SmallResNet(num_classes=100)` once.
- Keep original CIFAR-100 labels.
- Do not call `expand_head()`.
- Do not restart the model or optimizer between class groups.
- Train from scratch rather than loading the old task-trained checkpoints.

The required run uses classes 0–19 while retaining the planned 100-output head.

### 3. Connect to Khoi's Loader

- Use the shared mixed-class loader.
- Remove the separate `TASKS = [1, 2]` training sequence.
- Avoid keeping a second data pipeline inside the training script.
- Agree with Khoi on transforms, input scaling, and normalization.

### 4. Replace Repeated Task Training

The old notebook trains each task for 50 epochs.

For this run:
- Use batch size 32.
- Process the mixed training stream once.
- Apply one cross-entropy update per batch.
- Keep the final smaller batch.
- Count training examples and optimizer updates.
- Do not replay earlier examples.

Reuse the batch-update logic where possible.

Initialize the optimizer once. If keeping OneCycleLR, use the number of
batches in the single pass as its total steps, not 50 epochs.
Record all optimizer and scheduler settings.

### 5. Update Evaluation

Reuse the evaluation function with these changes:
- Evaluate on the combined validation subset.
- Do not use task-specific prediction masks.
- Evaluate before training, about every 1,000 examples, and at the end.
- If a batch crosses a threshold, evaluate after that batch and record
  the actual example count.
- Disable gradients and use evaluation mode.
- Verify that weights and BatchNorm statistics do not change.
- Restore training mode before continuing.

Replace task-accuracy tables and forgetting calculations with
accuracy-versus-examples records.

Keep official test data out of development and tuning.

### 6. Save Results and Run Details

- Write `results/online_results.csv` using the shared format below.
- Coordinate with Alka so her plotter reads it directly.
- Save the configuration, seeds, package versions, device, and exact command.
- Update checkpoint metadata to describe shuffled online learning.
- Do not retain the old `domain-incremental` metadata label.
- Keep checkpoints local.

### Tests

Check:
- The model returns 100 scores per image.
- A small batch completes a training update.
- Training stops after one pass.
- Example and update counters match the stream.
- Evaluation leaves weights and running statistics unchanged.
- The results CSV contains valid checkpoint records.

### Submit

- Model and shared evaluation files.
- Online training script and configuration.
- Test output.
- One completed 20-class online run.
- Results CSV and exact run command.

### Done When

Another member can run the experiment from the repo without editing the notebook.

## Alka — Adapt Plots, Results, and Documentation

### Build on Week 1

Reuse your result template and plotting code.
Review Nguyen's notebook plots with him and reuse useful pieces rather
than creating duplicate plotting tools.

### Tasks

1. Agree on the shared CSV format with Nguyen.
2. Update `results/results_template.csv`.
3. Update `src/plot_results.py` to read `results/online_results.csv`.
4. Replace task-based plots with:
   - X-axis: training examples processed.
   - Y-axis: validation accuracy.
5. Save the plot as `results/figures/online_learning_curve.png`.
6. Label the run, class count, and evaluation split clearly.
7. Handle missing files, empty results, and invalid values with clear messages.
8. Test with synthetic results while waiting for the real run.
   Label synthetic output as an example, not an experiment finding.
9. Update the protocol and checklist to remove required Task 1 → Task 2,
   EWC, and Fisher work.
10. Check setup and run commands with Khoi and Nguyen.
11. Write `docs/week2_results_summary.md` with:
    - What changed from Week 1.
    - Settings used.
    - Observed accuracy and training time.
    - Errors or unfinished work.
    - Next steps.
12. Keep old results clearly labeled as the earlier class-incremental experiment.
    Do not compare them directly with the new run without explaining the
    different training settings.

### Submit

- Updated result template and plotter.
- Plotting checks.
- Real learning curve.
- Short results summary.
- Updated checklist and run instructions.

### Done When

The plotter reads Nguyen's CSV without manual edits, and the summary
matches the saved results.

## Dong Quan Tran — Coordinate and Integrate

### Tasks

1. Share Dr. Athitsos's October 13 confirmation with the team.
2. Record any further changes he requests.
3. Make sure Week 1 submissions are available to everyone.
4. Settle shared interfaces, settings, and result format early.
5. Review pull requests and test evidence before merging.
6. Keep the README, protocol, direction document, and assignment consistent.
7. Mark older task-based instructions as historical.
8. Preserve completed contributions and avoid duplicate work.
9. Coordinate the integrated run and a fresh-checkout reproduction check.
10. Track blockers and arrange help.
11. Summarize effort, follow-through, and remaining work honestly.

### Submit

- Recorded advisor confirmation and agreed settings.
- Integration review and test evidence.
- Short team update.

# Shared Results Format

Save results to `results/online_results.csv`:

```csv
mode,run_id,model_seed,stream_seed,num_classes,examples_processed,updates,passes,batch_size,learning_rate,optimizer,eval_split,accuracy,training_seconds
```

Rules:
- One row per evaluation checkpoint.
- Use `online` for mode.
- Use `validation` for development evaluation.
- Accuracy is a decimal from 0 to 1.
- Count only training examples in `examples_processed`.
- `updates` counts optimizer steps.
- `passes` records the fraction of the training stream completed, ending at 1.
- `training_seconds` excludes evaluation time.
- Link each run to its saved configuration.

Record memory use separately if measured, including the measurement method.

# Order of Work

## Early Week

- Agree on settings and interfaces.
- Khoi adapts the stream.
- Nguyen moves reusable notebook code into the repo.
- Alka updates the CSV format and tests plotting.
- Tuong reviews the explanation and protocol.

## Midweek

- Complete a short integrated run.
- Check data separation, counters, evaluation, and CSV output.
- Report blockers before starting the full required run.

## End of Week

- Finish the 20-class single-pass run.
- Generate the real learning curve.
- Have another member run the documented commands.
- Submit the summary and list remaining problems.

# Stretch Goals

Only after the required work is complete:
- Expand to all 100 classes.
- Repeat with additional recorded seeds.
- Add a multiple-pass offline comparison.

For an offline comparison, keep the model, data splits, and transforms the same.
Report extra passes, updates, and time. Do not call it an equal-budget comparison.

EWC and Fisher estimation remain deferred.

# Progress Updates

Post a short midweek and end-of-week update:

```text
Completed:
Evidence:
Blocker:
Next step:
```

Evidence can include commits, test output, run logs, screenshots, shared drafts,
or review comments.

Raise blockers early. Asking for help, reviewing work, and correcting mistakes
count as useful progress.

Ratings prioritize effort, commitment, communication, teamwork, and follow-through.
Accuracy and GitHub activity alone do not determine a rating.

# End-of-Week Checklist

- [ ] Week 1 work is preserved and available.
- [x] Online-learning direction confirmed by Dr. Athitsos on October 13, 2026.
- [ ] Shared implementation settings are recorded.
- [ ] Classes 0–19 are combined before shuffling.
- [ ] Each training example appears once, with no dropped final batch.
- [ ] Training, validation, and test examples are separate.
- [ ] The classifier stays fixed throughout the run.
- [ ] Training uses one pass with no replay.
- [ ] Evaluation leaves the model unchanged.
- [ ] Data, training, evaluation, and plotting checks pass.
- [ ] Results CSV and real learning curve are saved.
- [ ] Another member can run the documented commands.
- [ ] The summary lists actual results and unfinished work.

Do not commit dataset images, checkpoints, secrets, or virtual environments.
