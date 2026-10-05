# Two-Week Assignment: Foundation for the Semester Project

**Project:** Online Learning – The Human Way  
**Dataset:** CIFAR-100  
**Duration:** Two weeks  
**Main focus:** Catastrophic forgetting and Elastic Weight Consolidation (EWC)

## Goal

During these two weeks, we will build a small and reproducible continual-learning experiment using CIFAR-100.

The required experiment is:

1. Train a model on Task 1, which contains classes 0–9.
2. Test its Task 1 accuracy.
3. Continue training the same model on Task 2, which contains classes 10–19.
4. Test Task 1 and Task 2 again.
5. Measure whether the model forgot Task 1.
6. Add standard EWC and compare it with the normal baseline.

The goal is not to implement EWC-DR yet. EWC-DR is an optional future extension after the normal baseline and standard EWC experiment are working correctly.

## Semester Connection

This two-week assignment builds the foundation for the rest of the semester project.

By the end of these two weeks, we should have:

- A shared and organized GitHub repository
- A verified CIFAR-100 task split
- A normal sequential-learning baseline
- A standard EWC implementation
- First baseline-versus-EWC results
- CSV result files and plots
- Documentation that can later be used in the final report and poster

After this assignment, we can expand from two tasks to more CIFAR-100 tasks, repeat experiments with multiple seeds, analyze the results, and prepare final deliverables.

## Shared Settings

| Item | Required Setting |
|---|---|
| Framework | Python and PyTorch |
| Dataset | CIFAR-100 |
| Model | One shared small CNN with 100 output classes |
| Learning setting | Class-incremental continual learning |
| Full task split | 10 tasks with 10 classes each |
| Current experiment | Task 1 and Task 2 only |
| Task 1 | Class IDs 0–9 |
| Task 2 | Class IDs 10–19 |
| Baseline | Standard cross-entropy loss only |
| EWC | Cross-entropy loss plus EWC penalty |
| Initial random seed | 42 |
| Required metrics | Per-task accuracy and Task 1 forgetting |

The baseline and EWC runs must use the same model, task split, image transforms, seed, optimizer, number of epochs, batch size, and learning rate. Only the EWC penalty and lambda value may differ.

---

# Week 1 — Baseline and Reproducible Setup

## Week 1 Goal

By the end of Week 1, the team must have a normal CNN that:

1. Trains on Task 1.
2. Is tested on Task 1.
3. Continues training on Task 2.
4. Is tested on both Task 1 and Task 2.
5. Saves checkpoints, results, and a forgetting plot.

This gives us the baseline evidence of catastrophic forgetting.

## Tuong — EWC Research Notes and Method Plan

### Purpose

Create a simple guide that explains EWC before the team implements it in Week 2. This guide can later be used for the background and methods sections of the final report.

### Tasks

1. Create `docs/ewc_guide.md`.
2. Explain catastrophic forgetting in simple words.
3. Explain why normal sequential training can hurt performance on older tasks.
4. Explain standard EWC in simple words:
   - Train on Task 1.
   - Save the model weights after Task 1.
   - Estimate which weights are important for Task 1.
   - During Task 2, discourage important weights from changing too much.
5. Include and explain this EWC equation:

```text
total_loss = new_task_loss + (lambda / 2) * sum(F_i * (theta_i - theta_star_i)^2)
```

6. Explain the meaning of:
   - `new_task_loss`: normal classification loss for the current task.
   - `theta_star_i`: saved value of a weight after Task 1.
   - `F_i`: importance of that weight for Task 1.
   - `lambda`: how strongly EWC protects old knowledge.
7. Add a simple EWC flow diagram:

```text
Train Task 1
→ Save Task 1 model weights
→ Calculate importance values
→ Train Task 2 with EWC penalty
→ Test Task 1 and Task 2
```

8. Add a short EWC versus EWC-DR comparison table.
9. Explain why this project will implement standard EWC first and treat EWC-DR as a possible later extension.
10. Cite the original EWC paper and the EWC-DR paper.

### Deliverables

- `docs/ewc_guide.md`
- EWC flow diagram
- EWC versus EWC-DR comparison table
- Paper references

### Progress Evidence

- Midweek: first draft committed or shared through a pull request.
- End of Week 1: completed guide merged into `main`.

---

## Khoi — CIFAR-100 Data Pipeline and Task Split

### Purpose

Build the shared CIFAR-100 data loader and fixed task split. Every experiment must use the same split so results are fair and reproducible.

### Tasks

1. Create `src/data.py`.
2. Load CIFAR-100 with Torchvision.
3. Download CIFAR-100 automatically if it is missing.
4. Create functions similar to:

```python
get_task_dataset(task_id, split="train")
get_task_dataset(task_id, split="test")
```

5. Use this fixed task split:

| Task | Class IDs |
|---|---|
| Task 1 | 0–9 |
| Task 2 | 10–19 |
| Task 3 | 20–29 |
| Task 4 | 30–39 |
| Task 5 | 40–49 |
| Task 6 | 50–59 |
| Task 7 | 60–69 |
| Task 8 | 70–79 |
| Task 9 | 80–89 |
| Task 10 | 90–99 |

6. Use `data/task_split.json` as the saved record of this split.
7. Print the class IDs, class names, and image counts for Task 1 and Task 2.
8. Save one image grid for Task 1 and one image grid for Task 2.
9. Create `tests/test_task_split.py`.
10. The test must verify:
   - Task 1 contains only labels 0–9.
   - Task 2 contains only labels 10–19.
   - Task 1 and Task 2 do not overlap.
11. Add short data setup instructions to the README.

### Deliverables

- `src/data.py`
- `tests/test_task_split.py`
- Updated `data/task_split.json` if needed
- `results/figures/task1_samples.png`
- `results/figures/task2_samples.png`
- README data instructions

### Progress Evidence

- Midweek: show Task 1 and Task 2 class names, labels, sample counts, and one sample-image grid.
- End of Week 1: another team member can load both tasks successfully.

---

## Nguyen — Baseline CNN and Sequential Training

### Purpose

Create the normal sequential-learning baseline. This baseline will be compared against EWC in Week 2 and later in the final report.

### Tasks

1. Create `src/model.py`.
2. Build a small CNN with 100 output classes.
3. Create `src/train_baseline.py`.
4. Include settings for:
   - Random seed
   - Number of epochs
   - Batch size
   - Learning rate
   - Device: CPU or GPU
5. Train the model on Task 1.
6. After Task 1 training:
   - Evaluate on Task 1 test data.
   - Save `checkpoints/baseline_after_task1.pt`.
   - Save the Task 1 result.
7. Continue training the same model on Task 2.
8. Do not restart the model.
9. Do not use Task 1 images while training on Task 2.
10. Do not use replay, EWC, or any forgetting-prevention method during the baseline.
11. After Task 2 training:
   - Evaluate on Task 1 test data.
   - Evaluate on Task 2 test data.
   - Save `checkpoints/baseline_after_task2.pt`.
12. Save results to `results/baseline_results.csv`.

Use this format:

```csv
method,lambda,trained_through_task,evaluated_task,accuracy,seed,epochs,batch_size,learning_rate
baseline,0,1,1,,,,,
baseline,0,2,1,,,,,
baseline,0,2,2,,,,,
```

### Deliverables

- `src/model.py`
- `src/train_baseline.py`
- `checkpoints/baseline_after_task1.pt`
- `checkpoints/baseline_after_task2.pt`
- `results/baseline_results.csv`
- README commands for running the baseline

### Progress Evidence

- Midweek: show a successful Task 1 training and evaluation run.
- End of Week 1: submit the three required accuracy values and checkpoints.

---

## Alka — Experiment Documentation, Results Format, and Baseline Plot

### Purpose

Make the project understandable, reproducible, and easy to present later in a report or poster.

### Tasks

1. Update `docs/experiment_protocol.md`.
2. Make sure it documents:
   - Research question
   - CIFAR-100 dataset
   - Task split
   - Model
   - Training order
   - Baseline definition
   - Metrics
   - Random seed
   - Hyperparameters
   - Saved outputs after each task
3. Review and update `results/results_template.csv` if necessary.
4. Create `src/plot_results.py`.
5. Create a baseline plot with:
   - Task 1 accuracy after Task 1 training.
   - Task 1 accuracy after Task 2 training.
   - Task 2 accuracy after Task 2 training.
6. Save the plot as `results/figures/baseline_forgetting.png`.
7. Add this short explanation to the result summary:

> If Task 1 accuracy drops after Task 2 training, the baseline model has forgotten part of Task 1. EWC will attempt to reduce that drop.

8. Update `docs/week1_checklist.md`.
9. Review the team files before the Week 1 meeting.

### Deliverables

- Updated `docs/experiment_protocol.md`
- Updated `results/results_template.csv`
- `src/plot_results.py`
- `results/figures/baseline_forgetting.png`
- Updated `docs/week1_checklist.md`

### Progress Evidence

- Midweek: submit an updated protocol and result-file format.
- End of Week 1: submit the completed baseline plot and checklist.

---

# Week 2 — Standard EWC and First Comparison

## Week 2 Goal

By the end of Week 2, the team must have a working standard EWC experiment and a fair comparison against normal sequential training.

We are not implementing EWC-DR this week. It stays as a possible future extension after the baseline and standard EWC experiments are stable.

## Tuong — Fisher Importance Guide and EWC Review

### Purpose

Document how EWC estimates which model weights are important for Task 1 and review the EWC implementation for correctness.

### Tasks

1. Create `docs/fisher_guide.md`.
2. Explain the diagonal Fisher approximation in simple words:

> Each model weight receives one importance score. A larger score means changing that weight may hurt Task 1 performance more.

3. Document the Fisher estimation process:

```text
Load model after Task 1
Load Task 1 training data
For each batch:
    Make predictions
    Calculate classification loss
    Run backpropagation
    Square each parameter gradient
    Add squared gradients to the importance estimate
Average the values at the end
```

4. Explain why model weights should not be changed while Fisher values are calculated.
5. Add these verification checks:
   - Each Fisher tensor has the same shape as its matching model parameter.
   - Fisher values are non-negative.
   - Task 1 data is used for Fisher estimation.
   - Task 2 data is not used for Fisher estimation.
6. Review the EWC code pull request and confirm that the code matches the standard EWC equation.

### Deliverables

- `docs/fisher_guide.md`
- Fisher verification checklist
- Pull-request review comments on EWC code

### Progress Evidence

- Midweek: Fisher guide draft committed or submitted as a pull request.
- End of Week 2: confirmation that EWC and Fisher calculations follow the documented method.

---

## Khoi — Fisher Data Loader and Reproducibility Checks

### Purpose

Make sure Task 1 data is used correctly for Fisher estimation while staying separate from Task 2 training.

### Tasks

1. Add a function similar to this in `src/data.py`:

```python
get_fisher_loader(task_id=0)
```

2. The Fisher loader must use Task 1 data only.
3. Confirm Task 1 images are not included in Task 2 training batches.
4. Extend data tests to verify:
   - Fisher loader labels are only 0–9.
   - Task 2 loader labels are only 10–19.
   - Fisher loader and Task 2 loader are separate.
5. Help run a small end-to-end EWC test:
   - One epoch on Task 1.
   - Calculate Fisher values.
   - One epoch on Task 2 with EWC.
6. Update README instructions for the data setup and EWC run.

### Deliverables

- Updated `src/data.py`
- Updated loader tests
- README reproducibility instructions
- Evidence that the small EWC test completed

### Progress Evidence

- Midweek: test output showing the Fisher loader uses only Task 1 labels.
- End of Week 2: a fresh clone can download data and complete the small test pipeline.

---

## Nguyen — Standard EWC Training and Lambda Experiments

### Purpose

Implement standard EWC using the shared CNN, task split, and training settings.

### Tasks

1. Create `src/fisher.py`.
2. After Task 1 training:
   - Save the Task 1 model parameters as `theta_star`.
   - Calculate Fisher importance values.
   - Save both items for use during Task 2 training.
3. Create `src/train_ewc.py`.
4. Use the same model and Task 1 training settings as the baseline.
5. Train Task 2 with this loss:

```text
total_loss = cross_entropy_loss + (lambda / 2) * sum(F_i * (theta_i - theta_star_i)^2)
```

6. Add an `--ewc_lambda` option.
7. Run these lambda values:

```text
0
10
100
1000
```

8. `lambda = 0` should act similarly to the baseline and is a useful debugging check.
9. For each lambda, save:
   - Task 1 accuracy after Task 2.
   - Task 2 accuracy after Task 2.
10. Save all results to `results/ewc_results.csv`.
11. Save the best balanced EWC checkpoint.

Use this format:

```csv
method,lambda,trained_through_task,evaluated_task,accuracy,seed,epochs,batch_size,learning_rate
ewc,0,2,1,,,,,
ewc,0,2,2,,,,,
ewc,10,2,1,,,,,
ewc,10,2,2,,,,,
ewc,100,2,1,,,,,
ewc,100,2,2,,,,,
ewc,1000,2,1,,,,,
ewc,1000,2,2,,,,,
```

### Deliverables

- `src/fisher.py`
- `src/train_ewc.py`
- `results/ewc_results.csv`
- EWC checkpoints
- README commands for running EWC

### Progress Evidence

- Midweek: complete one small EWC test run.
- End of Week 2: submit all required lambda results and checkpoints.

---

## Alka — Baseline versus EWC Results and Documentation

### Purpose

Turn experiment output into plots, tables, and documentation that can later be used in the report and poster.

### Tasks

1. Update `src/plot_results.py` so it reads:
   - `results/baseline_results.csv`
   - `results/ewc_results.csv`
2. Create these plots:
   - Baseline Task 1 accuracy before and after Task 2.
   - Baseline versus EWC Task 1 accuracy after Task 2.
   - EWC lambda versus Task 1 and Task 2 accuracy.
3. Save plots in `results/figures/`.
4. Create `docs/week2_results_summary.md`.
5. Include:
   - Best baseline result.
   - Best EWC result.
   - Lambda with the best Task 1 retention.
   - Whether that lambda lowered Task 2 accuracy.
   - Errors, limitations, or unexpected behavior.
6. Update `docs/decisions.md` with:
   - Why CIFAR-100 was chosen.
   - Why the first experiment uses two tasks.
   - Why EWC-DR is postponed.
   - Which EWC lambda values were tested.
7. Review the README so another student can reproduce a baseline and EWC run.

### Deliverables

- `results/figures/baseline_vs_ewc.png`
- `results/figures/lambda_tradeoff.png`
- `docs/week2_results_summary.md`
- Updated `docs/decisions.md`
- Updated README
- `docs/week2_checklist.md`

### Progress Evidence

- Midweek: draft comparison-plot script.
- End of Week 2: completed plots and a one-page result summary.

---

# Required Progress Updates

Each member must post one progress update around the middle of each week and another at the end of each week.

Use this format:

```text
Name:
Week:
Completed:
Evidence: GitHub pull request, commit, file path, screenshot, plot, or output
Blocker:
Next step:
```

A message such as “I am working on it” is not enough. Every update should include evidence of progress.

# End-of-Assignment Checklist

```text
[ ] CIFAR-100 downloads automatically.
[ ] Task 1 and Task 2 splits are verified.
[ ] Data tests pass.
[ ] Baseline CNN completes Task 1 → Task 2.
[ ] Baseline results and forgetting plot are saved.
[ ] Task 1 model parameters are saved.
[ ] Fisher values are calculated and checked.
[ ] Standard EWC completes Task 1 → Task 2.
[ ] Multiple lambda values are tested.
[ ] Baseline and EWC results are stored in CSV files.
[ ] Baseline and EWC comparison plots are saved.
[ ] README includes setup and run commands.
[ ] Documentation explains the experiment and EWC method.
[ ] No dataset files, checkpoints, secrets, or virtual environments are committed.
```

# What Happens After These Two Weeks

After this assignment is complete, the team will:

1. Expand the stable experiment from two tasks to more CIFAR-100 tasks.
2. Repeat final baseline and EWC experiments with multiple random seeds.
3. Report average accuracy, forgetting, runtime, and the retention-versus-new-task tradeoff.
4. Write the final technical report.
5. Finalize research notes and reproducibility instructions.
6. Create a poster or presentation using actual experiment results and plots.

Our main final deliverable is a controlled and reproducible comparison between normal sequential training and standard EWC. EWC-DR is optional future work only if the required experiments are complete.
