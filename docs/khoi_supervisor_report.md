# Khoi — final MNIST experiment report and supervisor Q&A

Date: 2026-10-30. Primary run: `mnist_nguyen_literal_seed42`.

## 1. Main result to present

I adapted the team's SmallResNet online-learning experiment to MNIST and
completed a fresh one-pass run using the requested 9,000 training / 1,000
validation examples. I matched Nguyen's published CIFAR training profile as
literally as practical, preserving all earlier experiments separately.

The final model achieved **96.5% validation accuracy: 965 correct, 35 errors**.
Macro precision was **96.5328%**, macro recall **96.5000%**, and macro F1
**0.964973**. These are final-checkpoint values, not selected peak values.
The official MNIST test set was not evaluated.

A separate checkpoint audit reloaded the saved weights and reproduced all
four metrics within 1e-12. This establishes internal consistency of the saved
result; it does not by itself establish generalization, multiple-seed robustness,
or fairness of every cross-dataset comparison.

## 2. Exactly what I contributed

- MNIST support with stable `train:<original row ID>` identities, seeded sample
  selection, fixed train/validation membership, one training shuffle, and no
  second DataLoader shuffle.
- Configurable subset size, including the required 9,000/1,000 experiment.
- Extracted reusable SmallResNet, one-pass trainer, evaluation and macro metrics.
- Preserved native-MNIST experiments while adding an optimizer-aligned variant
  and a separate literal CIFAR-profile variant.
- AdamW and OneCycle support, stepping once after each optimizer update; actual
  checkpoint learning rates, configured maximum LR and weight decay recorded.
- Deterministic evaluation without optimization or BatchNorm updates.
- Exact saved membership/order, configuration, environment, CSV, checkpoint
  fingerprint, per-sample validation predictions, confusion matrix and mistakes.
- Automated tests, independent real-checkpoint audit, and executed comparison
  notebooks. No downloaded dataset or model checkpoint is committed.

This demonstrates a reusable dataset adaptation and complete measured experiment.
It does not establish resistance to forgetting when tasks/distributions change.

## 3. Primary experiment settings

| Setting | Newest MNIST run |
|---|---|
| Source dataset | Official MNIST training source; original test source unused |
| Real labels | Ten digits, 0-9; labels not remapped |
| Selected rows | 10,000; remaining 50,000 training-source rows unused |
| Split | 9,000 training / 1,000 validation; balanced, no overlap |
| Per digit | 900 training / 100 validation |
| Membership procedure | NumPy default_rng seed 42; per-digit permutation; validation first, then training; sorted membership before stream permutation |
| Stream | One torch permutation with stream seed 42; each selected training row once |
| Input adaptation | Resize grayscale 28x28 to 32x32, bilinear/antialias; replicate into three channels |
| Network | Three-stage residual SmallResNet with 64/128/256 feature widths |
| Output head | 100 outputs; argmax over all outputs, no class mask |
| Dropout | 0.2 |
| Model / split / stream seeds | 42 / 42 / 42 |
| Train transform | RandomCrop(32, reflect padding 4), horizontal flip p=0.5, float32 scaling to [0,1] |
| Validation transform | Same deterministic input adaptation, then [0,1] scaling; no random crop/flip |
| Model normalization | CIFAR mean (0.5071,0.4865,0.4409), std (0.2673,0.2564,0.2762), applied once inside model |
| Optimizer | AdamW, PyTorch defaults except LR and weight decay |
| Learning rate | OneCycle maximum 0.0005, pct_start=0.15; other constructor defaults match Nguyen |
| Weight decay | 0.0005 |
| Training | Fresh initialization, one pass, no replay, cross-entropy |
| Batch / updates | 32 / 282; 281 full batches and final batch of 8 |
| Evaluation | Batch 256, two workers, same example checkpoints as Nguyen |
| Checkpoints | 0, 1024, 2016, 3008, 4000, 5024, 6016, 7008, 8000, 9000 |
| Device | CPU; CUDA availability was False |
| Determinism | torch deterministic algorithms; cudnn deterministic, benchmark disabled; CUBLAS config recorded in implementation |

The model has 100 possible outputs, not 100 actual MNIST classes. Accuracy counts
predictions into outputs 10-99 as wrong. At the final checkpoint there were zero
such predictions. Macro scores average over the ten real digit classes; averaging
90 nonexistent classes as zeros would be misleading. The full 100x100 confusion
matrix is retained; its plot aggregates unused outputs into an Other column.

## 4. Side-by-side team results

| Measure | Khoi: MNIST SmallResNet | Nguyen: CIFAR SmallResNet | Alka: CIFAR GaussianNB |
|---|---|---|---|
| Real classes | 10 digits | 20 selected object classes | Same 20 CIFAR classes |
| Training / validation | 9,000 / 1,000 | 9,000 / 1,000 | 9,000 / 1,000 |
| Per-class train / validation | 900 / 100 | 450 / 50 | Reported shared CIFAR split |
| Updates / batch | 282 / 32 | 282 / 32 | 282 / 32 |
| Final validation accuracy | **96.5%** | **43.2%** | **20.6%** |
| Macro precision | 96.5328% | Not exported | Not exported |
| Macro recall | 96.5% | Not exported | Not exported |
| Macro F1 | 0.964973 | Not exported | Not exported |
| Training/update time | 157.632s | 4.651s reported | 0.788s reported |
| Timed run wall time | 365.371s | Not exported | Not exported |
| Device | CPU | Tesla T4 CUDA, team report | CPU, team report |
| Independent local checkpoint audit | Yes | No | No |

Nguyen's neural profile is the reference for optimizer, scheduler, dropout,
100-output architecture, input format, normalization and augmentation settings.
Alka uses flattened 3,072 pixel features scaled to [0,1], without neural
augmentation/normalization, and updates one GaussianNB via partial_fit. A learning
rate, weight decay and dropout do not apply to GaussianNB.

Timing is not a speed ranking: Khoi's compute timer excludes loading/evaluation;
Alka times partial_fit only; Nguyen's notebook subtracts evaluation from elapsed
loop time, which includes other work and uses a GPU. No end-to-end speedup ratio
or optimizer-only speed claim should be made. Source hardware/version differences
are in run.json or the attributed team reports.

Why the per-class counts differ: 9,000/1,000 across ten MNIST classes requires
900/100 each, while twenty CIFAR classes require 450/50 each. Both splits are
balanced with a 90/10 ratio. Using 450/50 for ten digits would violate the requested
totals by producing only 4,500/500. Repeating grayscale into RGB does not add
information or change the count of true classes.

## 5. Source provenance: do not mix old and new CIFAR runs

The comparison notebook uses the updated combined CSV published on Alka's branch
at commit `10c9ef76da8df1de3ab97410df20c0c2f40b8258`, now also merged to main.
Its `resnet_nguyen_seed42` alias maps, per the handoff record, to Nguyen's
`online_c20_m42_s42_20261028-173041` run: max LR 0.0005, final accuracy 43.2%.
Nguyen's src notebook reference is commit
`3d356f9efa7bbcff1c07eb49936a53bf1692a8c0`.

The older `results/smallresnet-cifar-100.csv` is a DIFFERENT experiment:
`online_c20_m42_s42_20261015-200612`, LR 0.002, final 39.5%. It is preserved and
excluded from the new charts. Earlier notes saying no 43.2% CSV was available
are historical; the updated combined export is now available, with attribution.

Alka reports matching Nguyen's saved stream. The archived handoff explicitly
says the review has not independently reproduced that check. Original CIFAR
split/checkpoint artifacts are still needed for independent validation. I checked
export consistency and source fingerprints, not those models' actual predictions.

## 6. What testing actually proves

Fresh regression command on the submission code:

```powershell
$env:PYTEST_DEBUG_TEMPROOT=(Resolve-Path .pytest_cache).Path
$env:HF_HUB_OFFLINE='1'
$env:HF_DATASETS_OFFLINE='1'
python -m pytest -v --tb=short
```

Actual result: **35 passed in 23.11 seconds**; no pytest warnings reported.
Tests cover fixed membership, split separation, labels, source IDs, no repeats,
coverage, batch shapes/scaling, deterministic stream ordering, final batches,
stratified counts, literal input/head/normalization, training updates, evaluation
state preservation, metric arithmetic, unused-output prediction handling, full
282-step LR/beta1 schedule parity, and preservation of original Adam behavior.

Many MNIST contract tests use explicitly synthetic source images. They prove code
contracts, not accuracy. CIFAR tests use cached source data. The real training run
and real saved-checkpoint audit separately establish the reported MNIST metrics.

Checkpoint audit:

```powershell
python -m src.audit_mnist --run data/processed/mnist_nguyen_literal_seed42 --output results/mnist_nguyen_literal_seed42/audit
```

It directly indexes the saved validation source rows, loads the actual model,
checks 100 outputs without masking, and confirms:

- 9,000 unique training / 1,000 unique validation IDs; zero overlap.
- 900/100 examples per digit; configured loader membership/order matches saved IDs.
- 965 correct and 35 mistakes, all four final CSV metrics matching within 1e-12.
- Direct accuracy count and separate per-class macro calculations agree.
- Model weights and BatchNorm buffers unchanged during inference.
- No official test-set evaluation; no training during the audit.

The audit shares model/transform definitions with training, so it is an independent
checkpoint inference run, not a fully independent reimplementation of the pipeline.
It checks consistency; source-code review and tests remain necessary.

The three-way report notebook executed all five code cells and validated its
CSV fingerprint, run aliases, checkpoints and core settings. Charts were visually
checked. It does not rerun Nguyen/Alka's tests or training.

## 7. Error analysis

The newest model made 35 errors. Digit 3 had the most errors: 8/100; digit 5 had
6/100. The largest single confusion was true 3 predicted as 5 (5 examples).
Other frequent confusions included 5 -> 3 and 2 -> 6 (3 examples each).
These are descriptive observations, not proof that augmentation caused them.

The notebook includes the full confusion summary and the first 20 mistakes in
saved validation order. They were not cherry-picked by appearance. The checkpoint
fingerprint and all 1,000 labeled predictions are archived for audit.

## 8. Why the result looks good — and what I cannot claim

The model learned recognizable digit structure under the one-pass limit.
Convolutions exploit local spatial patterns and residual connections support
optimization. MNIST is relatively structured compared with CIFAR: centered digits,
grayscale source images and generally simple backgrounds versus varied color
objects/backgrounds and more selected classes. These are plausible explanations
for the gap, not isolated causes demonstrated by these experiments.

I should say: **"The MNIST experiment learned effectively and the result is
reproducible from its saved checkpoint."**

I should NOT say: **"My model is 53.3 percentage points better than Nguyen's
method"**, or that my work is superior because 96.5 exceeds 43.2. The validation
tasks differ. GaussianNB is a useful lightweight baseline even if its score is
lower; its feature assumptions/pipeline differ from the CNN.

The original optimizer-only MNIST comparison is more controlled: same random
membership/order, head, preprocessing and model seed; Adam 0.001 yielded 96.4%,
while AdamW+OneCycle+weight decay yielded 98.8%. The difference is 2.4 percentage
points on that shared validation set, but three optimizer components changed
together and only one seed was run. It does not isolate AdamW, LR or decay alone.

The newest literal run is 96.5%, not the highest score we observed. It changes
preprocessing, normalization, head/input and membership as well as optimizer
settings, so its difference from 98.8% cannot be attributed to a single cause.
The 0.1-point difference from the original 96.4% is not evidence of meaningful
improvement: validation memberships differ, and 0.1 point is only one of 1,000
images. We present the requested protocol, not whichever score looks best.

## 9. Historical runs — label them accurately

| Run | Train / validation | Protocol | Final validation accuracy |
|---|---|---|---|
| mnist_seed42_complete | 54,000 / 6,000 | Native MNIST; Adam 0.001 | 99.0833% |
| mnist_9000_seed42 | 9,000 / 1,000 | Native MNIST; Adam 0.001 | 96.4% |
| mnist_nguyen_optimizer_seed42 | Same random 9,000/1,000 membership/order as preceding run | Native MNIST; AdamW+OneCycle+decay | 98.8% |
| mnist_nguyen_literal_seed42 | New stratified 9,000 / 1,000 | Literal CIFAR-style input/head/transforms/normalization/optimizer | **96.5% — primary requested result** |

All runs began from fresh initialization; none resumed from the preceding run's
weights. This is not one model trained for four passes. However, repeated
inspection of validation results can still influence experiment selection.
Retaining the entire run history makes that process transparent.

## 10. Likely supervisor questions and concise answers

**Did you use the exact CIFAR settings?**  
I matched the referenced src notebook's architecture/input format/head,
normalization, augmentation, optimizer, schedule, dropout, seeds, batch sizes,
workers and checkpoints. MNIST source adaptation, real class count/source rows,
CPU fallback, and environment differences remain explicit. I also did not repeat
the notebook's train-mode dummy shape forward that can alter BatchNorm buffers.

**Why not keep MNIST at one channel and ten outputs?**  
That is appropriate for the native-MNIST baseline, which is preserved. This separate
variant intentionally copies Nguyen's 3-channel/100-output interface. Grayscale
is resized/copied, not turned into new color information.

**Why use CIFAR mean/std on MNIST?**  
To follow the literal-comparison request, not because those are MNIST statistics.
Normalization occurs once in the model. A dataset-specific normalization ablation
would be a separate experiment and should not silently replace this one.

**Could flips corrupt the meaning of a digit?**  
Yes, that is a methodological limitation. The transform is copied intentionally.
High accuracy does not prove the transform is harmless. A fixed-membership
flip/no-flip ablation would test its effect more directly.

**Why 900/100 per class instead of 450/50?**  
MNIST has ten classes, not twenty. Both keep 9,000/1,000 totals and a balanced
90/10 ratio. Matching both totals and per-class counts is mathematically impossible
without changing the real task/classes.

**Was there data leakage?**  
Saved source IDs show no training/validation overlap. Validation never enters the
optimizer loop. Official test data was unused. These checks do not eliminate the
separate issue of repeated validation-based decision making.

**Is it overfitting because you ran several times?**  
Runs restarted from scratch, so weights did not accumulate passes. Nevertheless,
repeated validation inspection can produce selection bias. We have not measured
train-versus-validation accuracy gaps or official-test performance, so absence
of overfitting is not established. Alignment was requested externally, but this
does not remove the need to report the full history and limitations.

**Why not report 98.8% or 99.08% instead?**  
Those belong to other protocols/sample counts. The result for the newest requested
literal profile is 96.5%. Every earlier result is preserved with its own identity.

**Why is MNIST so much higher than CIFAR?**  
Different image content/difficulty and true class count are plausible reasons.
It is a cross-dataset observation, not a model superiority result. Multiple seeds
and a controlled protocol are needed for stronger claims.

**Why is your runtime slower?**  
CPU versus Tesla T4 and different timing scopes make direct speed claims invalid.
The 365.371s wall time includes validation/loading/worker overhead during the run;
157.632s measures training computation only. Dataset setup precedes the wall timer.

**Does 100 outputs mean the model trained on 100 classes?**  
No. Only labels 0-9 occur. All 100 logits compete in argmax/cross-entropy;
outside-digit predictions count as wrong. Macro metrics use ten true labels.

**Did you independently verify Nguyen and Alka?**  
I checked their exported data and provenance and the notebook configuration.
I did not reload their original checkpoints. Their source-order match is a team
report, not an independent audit by me.

**How reproducible is this?**  
Exact rows/order, seed/config, versions, hardware, commands, CSV and checkpoint
hash are recorded. The saved checkpoint reproduces the final metrics here.
Cross-device/worker/library changes can still affect retraining results; bitwise
cross-platform retraining is not promised. Repeating training with seed 42 in the
same setup is a useful extra check, not claimed to have already been done.

**Is this continual-learning progress?**  
It completes a portable single-pass online-learning experiment on another dataset.
Randomly mixed fixed classes do not demonstrate resistance to catastrophic
forgetting when tasks/distributions change.

## 11. What to attach/show and what is next

Show `notebooks/team_mnist_cifar_comparison.ipynb` first: settings table, results,
three separate learning curves and an overlay, timing caveats, critical discussion
and pinned source links. For verification, show
`notebooks/mnist_nguyen_literal_comparison.ipynb`, its confusion matrix and audit.
The optimizer-only notebook explains the controlled MNIST settings comparison.

Attach/reference the newest config, result CSV/run.json/membership, audit JSON,
per-sample predictions, figures, exact test output summary and reproduction
commands. Raw source images and model.pt stay local; small mistake illustrations
are report figures, not a dataset upload. Anyone reproducing can retrain to a new
output folder; they cannot rerun the checkpoint audit without the local weights
or a separately shared checkpoint.

Next research steps, not completed claims:

1. Obtain original CIFAR split/run records/checkpoints; independently verify
   membership/order and export the missing precision/recall/F1.
2. Freeze settings before a team-approved one-time official test evaluation.
   Keep that test out of iterative development.
3. Repeat across agreed seeds and report mean/spread, retaining memberships.
4. Use fixed membership for augmentation/normalization/optimizer ablations.
5. If requested, run Khoi's original settings on CIFAR as a separate labeled
   experiment. No such reverse-transfer run has been performed here.

## 12. Reproduction commands

Use fresh output directories; do not overwrite completed experiments.

```powershell
python -m pip install -r requirements.txt
python -m pytest -v --tb=short
python -m src.train_online --config configs/mnist_nguyen_literal.yaml --output data/processed/mnist_nguyen_literal_seed42
python -m src.audit_mnist --run data/processed/mnist_nguyen_literal_seed42 --output results/mnist_nguyen_literal_seed42/audit
```

Notebook execution additionally needs nbformat, nbclient and ipykernel; use Run All.
See `results/team_comparison_sources/provenance.json` for pinned team sources.

## Short message to the supervisor (Vietnamese)

Anh ơi, em đã chạy lại MNIST với setting giống bản CIFAR của Nguyen nhất có thể:
AdamW, OneCycle max LR 0.0005, weight decay 0.0005, dropout 0.2, seed 42,
input 3x32x32, head 100 outputs, crop/flip và CIFAR mean/std. Tổng dữ liệu là
9.000 train / 1.000 validation, một pass, không replay. Kết quả cuối là 96.5%,
macro F1 khoảng 0.965; load lại checkpoint được đúng 965/1.000 ảnh. Em đã kiểm
tra không overlap và 35 tests đều pass. Máy em không có CUDA nên chạy CPU.
Em giữ các run cũ riêng và có notebook so sánh với Nguyen 43.2% và Alka 20.6%.
Em không kết luận model của em tốt hơn vì MNIST khác CIFAR về dữ liệu và số class;
phần timing cũng khác hardware và cách đo. Những giới hạn này đều ghi trong report.
