| Member | Credit as completed | What is still missing |
|---|---|---|
| Alka | CSV template, learning-curve plotter, successful Colab test with five synthetic checkpoints, and results-summary draft | Real CSV check, real plot, updated findings, reusable tests |
| Khoi | Mixed 20-class stream, seeded ordering, batch handling, and six written tests | Shared validation split, sample IDs, training integration, actual test evidence |
| Nguyen | Working online notebook, split and stream checks, evaluation-state checks, completed run, CSV/config/checkpoint saving, and plots | Publish the online work, connect shared loader, package runnable files, fix the shape check |
| Tuong | Online guide, EWC deferral note, and five-point review checklist | Correct and shorten the guide, apply the checklist to actual code, record review findings |

# Week 2 Wrap-Up Tasks

## 1. Khoi — Finish the Loader Integration

### Remaining tasks
- Work with Nguyen to use ONLY ONE shared training/validation split.
  - Reuse his existing 9,000/1,000 split and saved membership unless the team agrees otherwise.
- Return `(image, label, sample_id)` to match Nguyen’s training interface.
- Keep sample identities clear: source split plus row ID.
- Match the agreed transforms and scaling. Avoid double normalization.
- Extend the existing tests to check:
  - No training/validation overlap.
  - Every selected training sample appears exactly once.
  - Correct sample IDs and final-batch size.
- Update the current 10,000-sample expectations if the loader now reserves validation data.
- Run the real loader tests and attach the command and output with Pull Request #7.

Finish when: Nguyen uses this loader directly, with no duplicate notebook data pipeline.

## 2. Nguyen — Publish and Package the Existing Experiment

### Remaining tasks
- Upload to everything to github under Nguyen's branch: the online notebook and its existing CSV, configuration, run records, and split/order metadata.
- Keep the original 39.5% run unchanged and clearly identified.
- Move reusable code into model, training, and evaluation files.
- Replace duplicate data-loading code with Khoi’s shared loader.
- Fix the initial shape check so it does not update BatchNorm statistics.
- Make the saved configuration the source of settings, rather than maintaining separate notebook values.
- Add small, reusable checks for model output shape and a training update.
- After integration, run the final pipeline once and save it under a new run ID.
- Provide exact setup, test, and training commands.

Finish when: another member can run the online experiment without editing notebook cells -> Reimplementation.

## 3. Alka — Use the Real Results

### Remaining tasks
- Get Nguyen’s actual CSV and run the repository plotter on it.
- Confirm no manual column changes are needed.
- Save the real learning curve with the run ID and evaluation split identified.
- Update the summary with:
  - Actual settings and sample counts.
  - Final validation accuracy.
  - Training time, if recorded.
  - Known limitations and remaining issues.
- Turn the existing manual checks into reusable tests, including the incomplete-row case we fixed.
- Confirm the earlier synthetic rendering check still works with the revised plotter.
- Check the final setup and plotting commands.

Finish when: the real plot and summary match the published CSV. No need to recreate Nguyen’s per-class plots if redundant.

## 4. Tuong — Correct the Guide and Complete the Actual Review

### Remaining tasks
- Shorten the guide to the explanation the team needs.
- Correct the memory claims: no replay does not mean the whole notebook uses constant memory.
- Remove the claim that discarding data automatically establishes privacy-law compliance.
- Remove “upper bound” language for one run’s final accuracy.
- Clarify that the 1% and 5% lines describe different uniform-guessing rules—not guaranteed accuracy for a randomly initialized network.
- Correct the explanation of `model.train()` versus gradient recording.
- Update the reported “open choices” using the team’s actual agreed configuration.
- Apply the five-point checklist to Khoi’s and Nguyen’s final changes.
- Record file-specific findings and unresolved issues in the PR reviews.
- Help Alka write the limitations section.

Finish when: the guide matches the code, and the review contains actual findings, not only a checklist.

## All members aim for

- One shared loader and training pipeline.
- Real test output.
- Github: published online notebook/code and run records.
- A real CSV-to-plot check.
- Corrected method notes and completed code review.
- One successful fresh-checkout reproduction.
