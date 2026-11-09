# Test-Time Training: Team Assignment

## Goal

Implement two separate experiments approved by Dr. Athitsos. Both use rotation prediction to adapt to test images without using their class labels.

- Experiment A, Nguyen: Train on CIFAR-10 and evaluate on CIFAR-10-C.
- Experiment B, Tuong: Use CIFAR-100 classes 0–19 and compare frozen models, standard TTT, and online TTT on clean and noisy images.

These experiments answer different questions. Keep their settings and results separate. Shared tools are welcome, but do not directly rank results across different datasets.

## 1. Nguyen: Lead Experiment A

### Question

Does rotation-based test-time training improve classification on corrupted CIFAR-10 images?

### Tasks

1. Prepare separate source-training, development, and evaluation data. Document the split.
2. Build a shared backbone with a classification branch and a four-class rotation branch: 0, 90, 180, and 270 degrees.
3. Train a supervised-only model and a joint model using classification plus rotation loss.
4. Save checkpoints, training settings, seeds, and source-training results.
5. Implement test-time rotation updates. Update only the agreed shared backbone; keep both heads frozen.
6. Start with one agreed CIFAR-10-C corruption and severity, plus clean images. Expand only after the small run works.
7. Compare supervised frozen, joint frozen, and the chosen TTT mode. State whether updates reset or carry forward.

### Deliverables

- Runnable training and adaptation code.
- Saved checkpoint locations and exact run commands.
- A results CSV and a short summary showing whether TTT beats joint frozen.
- A note explaining failures or setup limits, if any.

## 2. Tuong: Lead Experiment B

### Question

Does adaptation help our existing twenty-class setup, and is keeping updates better than resetting them?

### Tasks

1. Use CIFAR-100 classes 0–19 with a fixed, saved split.
2. Implement the proposed shared backbone and classification/rotation branches. Use GroupNorm consistently in matched baselines.
3. Train a supervised-only model and a jointly trained model. Record source-training budgets separately from adaptation costs.
4. Retain the proposed 100-output classification head and state that evaluation uses classes 0–19. Do not silently mask outputs.
5. Compare four conditions:
   - Supervised frozen: classification-only training, no test updates.
   - Joint frozen: joint training, no test updates.
   - Standard TTT: adapt, predict, then reset before the next image.
   - Online TTT: adapt, predict, and keep updates for the next image.
6. Start joint frozen and both TTT modes from the same joint checkpoint.
7. Test on clean images and one fixed Gaussian-noise condition. Record noise strength and how noise is applied.

### Deliverables

- Runnable code for all four conditions.
- Checkpoints, settings, and exact run commands.
- A results CSV comparing accuracy, runtime, and update counts.
- A short answer: did adaptation help, and did keeping updates help or hurt?

## 3. Khoi: Own Data and Evaluation Checks

### Question

Are both experiments testing what they claim, without using test labels during adaptation?

### Tasks

1. Work with each lead to save dataset membership, image order, seeds, and preprocessing settings.
2. Prepare repeatable clean and corrupted inputs for each experiment.
3. Keep development data separate from final evaluation data. Check for overlap.
4. Give the adaptation function images only. Keep class labels in the scoring code.
5. Check that frozen models do not change, standard TTT resets correctly, and online TTT keeps updates.
6. Check that each test image receives exactly one recorded class prediction per condition.
7. Check that independent runs reset model and optimizer state.
8. Run a small end-to-end check for each experiment and report problems before full runs.

### Deliverables

- Data preparation and evaluation-check code.
- A pass/fail checklist for each experiment.
- Saved split/order records or clear instructions to reproduce them.
- A short list of unresolved issues.

## 4. Alka: Own Results and Comparison

### Question

Does adaptation improve accuracy enough to justify its cost, and does it hurt clean-image performance?

### Tasks

1. Agree on one CSV format with both experiment leads.
2. Log experiment, condition, seed, checkpoint, dataset, corruption, severity/noise strength, accuracy, runtime, and update count.
3. Track rotation loss separately. Lower rotation loss does not by itself show better classification.
4. Report each adaptive condition's accuracy change in percentage points against its matched joint-frozen baseline.
5. For online TTT, plot accuracy over the stream using fixed-size windows.
6. Summarize clean and corrupted results separately. Never average the two experiments together.
7. After the small runs pass checks, summarize three seeds with mean and standard deviation.
8. Keep Tent/EATA as possible later work. Do not add them to this week's required scope.

### Deliverables

- Shared logging format and summary script.
- Separate result tables and plots for Experiments A and B.
- A short findings note covering gains, harm, runtime, and limits.

## Shared Rules

- Never use evaluation class labels for adaptation or choosing settings.
- Choose settings on development data, then lock them before final evaluation.
- Joint training must happen before rotation-based adaptation. Do not attach an untrained rotation head to an old classifier and call it prepared TTT.
- Within each experiment, match test images, order, preprocessing, and seeds across conditions.
- Disable dropout during evaluation. State which parameters can change.
- Record adaptation learning rate, steps, batch size, loss weights, prediction timing, and reset rules.
- Keep source-training time separate from adaptation time. Record hardware and timing scope.
- Save code version, configuration, seed, and results for each run.
- Negative results are valid. Report them instead of changing settings on final test data.

## Work Order

1. Agree on settings and interfaces. Nguyen and Tuong publish their experiment plans; Khoi checks evaluation rules; Alka defines the CSV fields.
2. Run a smoke test. Use a small subset and one seed to confirm training, adaptation, scoring, and saving work.
3. Review checks. Fix label access, reset, data, or logging problems before full runs.
4. Run the pilot. Complete each experiment's agreed conditions with one seed.
5. Repeat. If the pipeline passes checks, run three seeds and prepare the final comparison.

## Settings to Confirm Before Full Runs

- Source model size, training budget, optimizer, and loss weights.
- Development split and final evaluation set.
- Experiment A: first corruption, severity, and TTT reset mode.
- Experiment B: Gaussian-noise strength and application method.
- Adaptation learning rate, steps, batch size, and optimizer reset behavior.
- Sample count, seeds, hardware, and expected runtime.

## Done When

- Both experiments have reproducible run commands.
- Khoi's checks pass, or any remaining limits are clearly stated.
- Results include matched frozen baselines and clean controls.
- Alka can rebuild the tables from saved CSV files.
- Each lead explains what helped, what hurt, and what to try next.
