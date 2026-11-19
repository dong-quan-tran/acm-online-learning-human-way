# Fixed held-out TTT evaluation

The final protocol for Experiment A is Nguyen's original **8,000 adaptation /
2,000 fixed quiz** split. Experiment B remains **1,500 / 500**. The former A
6,000/2,000 draft and its smoke test are historical. Neither is required for
this review.

| Experiment | Source train | Source development | Online adaptation | Fixed quiz |
| --- | ---: | ---: | ---: | ---: |
| A · CIFAR-10 / CIFAR-10-C | 45,000 | 5,000 | 8,000 | 2,000 |
| B · CIFAR-100 classes 0–19 | 9,000 | 1,000 | 1,500 | 500 |

Use the same source ID for each image's clean and noisy views. Each online
stream starts separately from the joint source checkpoint. Score both views
before adaptation and after every 100 adaptation images. Continue the same
model and optimizer after scoring. Take a last measurement after a final
partial interval if there is one.

## Final checklist

PASS rows specify which evidence supports the check. A fixture tests the
code with a small synthetic model and the real ID lists. It does not
reproduce the trained results.

| Check | Nguyen A | Tuong B |
| --- | --- | --- |
| Target membership and zero source-ID overlap | **PASS.** Reconstructed the published 8,000/2,000 split exactly from the notebook code. | **PASS.** Published 1,500/500 file matches Khoi's saved manifest. |
| Same clean and corrupted source IDs | **PASS in code.** Both views index the same original rows. CIFAR-10-C labels are checked against official test order. | **PASS.** Exact same ordered IDs verified in both views. |
| Prior use for tuning and source checkpoint selection | **FAIL for untouched final-test status.** Earlier TTT sweeps scored these 2,000 IDs. The latest run uses fixed defaults. Source checkpoints are selected on a separate 5,000-image training-split development set. No path from quiz labels to checkpoint selection was found. | **PASS for current-run separation.** The published audit records no source-train or development overlap. The quiz reuses previously evaluated images, as disclosed. |
| Schedule | **PASS in code and fixture.** All 81 checkpoints per stream from 0 to 8,000. **UNVERIFIED for all trained rows.** The visible saved table has 0, 4,000 and 8,000 and the notebook displays the curve. The underlying CSV was unavailable. | **PASS with trained weights.** Pilot at 0/100/200/205 and full run every 100 through 1,500. |
| Exactly one prediction per fixed quiz ID per checkpoint and view | **PASS in fixture** using the reconstructed 2,000 IDs in order. **UNVERIFIED for the trained run.** The held-out monitor saves aggregate scores and the exported notebook has no per-ID held-out predictions. | **PASS with trained weights.** Audited 10,000 pilot and 34,000 full prediction records. |
| Scoring preserves weights, normalization buffers, optimizer and adaptation RNG | **PASS in CPU fixture.** Prior real-image CPU smoke checks with random weights also found unchanged state. **UNVERIFIED on the published trained GPU run.** | **PASS with trained weights.** Saved local checks passed for the pilot and full run. |
| Same starting checkpoint, independent stream runs and carry-forward | **PASS in code and fixture.** Both streams receive the same joint model, copy it, and start a new optimizer. One optimizer per online stream persisted through 8,000 steps in the fixture. **UNVERIFIED as an independent trained replay.** | **PASS with trained weights.** The local runs used the same joint checkpoint for both streams and carried updates through 205 and 1,500 images. |
| Trained accuracy | **Reported in saved notebook. Independent reproduction UNVERIFIED.** | **PASS for evaluation QA.** Local CPU replay and published GPU results show the same noisy-stream trade-off. Exact scores can differ. |

## Nguyen's split and settings

All 12 code cells in the public [Kaggle version 2](https://www.kaggle.com/code/vunguyenledinh/notebook9e2b34c05b)
match the [executed GitHub notebook at 64ae8d1](https://github.com/dong-quan-tran/acm-online-learning-human-way/blob/64ae8d1/notebooks/nguyen-expa.ipynb).
The [reconstructed manifest](../results/heldout_verification/nguyen-original-split.json)
contains the actual IDs implied by the split code, in the order used by the
online runner. It is a reconstruction, not the original run's splits.npz.

- Cells 2 and 4 (zero-based) use NumPy RandomState(0). They permute the
  50,000 official training rows first, then the 10,000 official test rows.
  The first 2,000 test permutation entries become sorted quiz rows. The
  remaining 8,000 become the sorted adaptation pool. Cell 16 applies
  RandomState(1).permutation(8000) to that pool. Both streams use that order.
- Cell 4 pairs clean CIFAR-10 with Gaussian noise severity 5 using the same
  test row and checks the corruption label block against the clean labels.
  The input is RGB CHW scaled by 255, followed by model normalization.
- The [earlier notebook at f2aa445](https://github.com/dong-quan-tran/acm-online-learning-human-way/blob/f2aa445/notebooks/nguyen-expa.ipynb)
  uses the same 2,000 source IDs in TDEV. Cell 14 compares online learning
  rates on all 2,000 and episodic learning rates and step counts on a
  500-image prefix. The saved winners were online 0.003 / one step and
  episodic 0.01 / ten steps. The adopted run used 0.001 for both with the
  same step counts. The historical exposure is confirmed. The current code
  has no sweep and there is no evidence it selected settings using the
  monitor. The earlier exposure limits claims about an untouched test set.
- Cells 8 and 10 choose the starting checkpoint by classification accuracy
  on the separate source development images. The saved log shows 30 source
  epochs for each model. No evidence shows the test-derived quiz selected
  an epoch, stopped training, or entered the source loss.
- In cells 12, 14 and 16, each online call deep-copies the joint model,
  initializes one optimizer, and calls the monitor before updating and
  after each 100 images. The monitor only calls predict and grades its
  output. Predict uses no_grad and eval mode. GroupNorm has no running
  averages. Dropout is off during scoring. The online path does not reset
  after a monitor call. One update per adaptation image, augmentation seed
  100, 32 rotation examples, and SGD momentum 0 are recorded for this run.

### Trained scores reported in Nguyen's notebook

| Adaptation stream | Fixed 2,000-image quiz | At 0 | At 4,000 | At 8,000 |
| --- | --- | ---: | ---: | ---: |
| Clean | Clean | 91.15% | 91.15% | 91.40% |
| Clean | Noisy | 40.80% | 38.50% | 41.80% |
| Noisy | Clean | 91.15% | 81.95% | 79.10% |
| Noisy | Noisy | 40.80% | 76.35% | 75.35% |

The **72.45%** online result on noisy adaptation-stream images scores each
of the 8,000 images after its own update. The **75.35%** score is on the
separate, fixed noisy quiz at the end of the same stream. Report them
separately and identify the quiz as the reused TTT development set.

### Repeat the audit

The [audit script](../results/heldout_verification/audit_nguyen.py) reads the
two pinned GitHub revisions and the Kaggle API code export. It executes only
the reviewed prediction, monitor and online runner definitions with a small
fixture. It traces the reconstructed IDs, checks every checkpoint and
compares weights, buffers, optimizer, and PyTorch/NumPy/Python RNG before and
after scoring. The [actual output](../results/heldout_verification/nguyen-notebook-audit.txt)
also captures the notebook's reported scores and settings.

    python results/heldout_verification/audit_nguyen.py --kaggle-export data/processed/heldout-evidence/kaggle-pull-current

GitHub and local artifacts provided the executed notebook but no Nguyen
checkpoint, complete heldout_curve.csv, or trained state audit. The public
Kaggle source export was accessible. The output page was inaccessible and
the two public output routes checked earlier returned HTTP 404. The
notebook lists generated files under /kaggle/working, which does not make
them available locally. Cell 16 writes predictions.npz for the adaptation
stream and heldout_curve.csv for aggregate quiz scores. Even the former
file would not by itself show per-ID quiz predictions. Full trained
verification requires the checkpoint for a replay, or a complete trained
curve plus ordered quiz predictions and state evidence. No retraining was
performed.

## Tuong's evidence

Tuong's code and split come from [revision 1df2e97](https://github.com/dong-quan-tran/acm-online-learning-human-way/tree/1df2e97).
Use the input manifest data/ttt_exp_b_held_out_draft.json in his revised
runner. The trained CPU runs used seed 42, Gaussian sigma 0.08, the
published checkpoints, and the [saved configuration](../results/heldout_verification/tuong-config.yaml).
For the 205-image check, set --max-adapt-images 205. The full stream
uses all 1,500 rows.

The [pilot](../results/heldout_verification/tuong-pilot.csv) and
[full replay](../results/heldout_verification/tuong-full.csv) include the
score rows, with matching
[pilot checks](../results/heldout_verification/tuong-pilot_checks.json) and
[full checks](../results/heldout_verification/tuong-full_checks.json).
The [verification log](../results/heldout_verification/tuong-verification.txt)
records the ordered-ID and prediction-count checks against the raw local
logs. The shared copies record the executed source revision instead of the
parent checkout revision detected by the exported runner. In the local
full noisy adaptation run, noisy quiz accuracy moved from 34.6% to 47.8%
and clean quiz accuracy moved from 65.2% to 50.0%.

The final scoped repository test run logged **45 passed in 37.83s**.
GaussianNB was excluded because scikit-learn was unavailable. The initial
unscoped collection also picked up duplicate tests from archived checkouts.
See [pytest output](../results/heldout_verification/pytest.txt).

The handoff has both manifests, loading instructions and the evidence for
each verdict. Experiment B has trained sign-off. Experiment A meets the
adopted split, with documented development reuse and trained checks that
remain unverified.
