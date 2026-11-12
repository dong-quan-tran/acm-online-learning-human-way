# TTT evaluation handoff

Use the same ordered source IDs for clean and corrupted conditions within each
experiment. For example, CIFAR-10 clean test row 17 and all its CIFAR-10-C
variants use the same ID, `test:17`. Keep the two datasets in separate manifests.
Do not rename corruption variants to make an overlap check pass.

## How to run the checks

1. Pass the agreed development IDs and ordered final evaluation IDs to
   `save_manifest` in `src.ttt_evaluation`, with the dataset, seed, preprocessing
   and corruption settings. Save the manifest under the run directory.
2. Make an ID-to-image mapping for the pilot. For A, `cifar10c_image` loads the
   matching official row for the chosen severity, scaled to CHW [0, 1]. For B,
   call `gaussian_noise` on CHW [0, 1] images before model normalization,
   recording its seed and standard deviation. Apply the agreed normalization
   consistently to clean and corrupted variants.
3. Pass the ordered IDs and image mapping to `evaluate_stream`. The callback
   signatures are `adapt(model, optimizer, image)` and `predict(model, image)`.
   Adaptation uses rotation labels generated from the image, never class labels.
   Keep the class-label mapping outside both callbacks and use it only in
   `score_predictions`. The runner does not sandbox a callback that separately
   imports labels: review the actual adapters before running them.
4. Use `frozen` for each frozen baseline and `standard` / `online` as agreed.
   Use the same pristine joint checkpoint and optimizer configuration for the
   joint frozen and adaptive conditions. Supply an optimizer bound to the
   supplied model, with fresh state for an independent run. Standard resets
   model and optimizer after each prediction. Online carries both forward.
   The snapshot checks are intended for small pilots; measure model runtime
   separately from this audit overhead.
5. Run both leads' actual adapters on small clean and corrupted subsets.
   Save ordered predictions and record failures before starting full runs.

## Checklist

| Check | A: CIFAR-10 / CIFAR-10-C | B: CIFAR-100 classes 0-19 |
| --- | --- | --- |
| Source-ID overlap / saved order | Pass: notebook's 2,000 development / 8,000 evaluation rows are disjoint | Pass: published 9,000 train / 1,000 development / 2,000 test rows checked |
| Corrupted input preparation | Pass: official Gaussian noise severity 5 aligned to clean labels | Pass: published clean/noisy order and repeatable noise at sigma 0.08 |
| Label-free callback and one prediction per image | Pass in corrected review copy; original published notebook still has both findings | Pass on inspected adapter and valid real-image stream |
| Frozen invariance; standard reset; online carry | Pass with randomly initialized model | Pass with published trained checkpoints |
| Independent model and optimizer state | Base model unchanged on rerun; new optimizer per call by inspection | Pass: fresh checkpoint copy per condition; stateless SGD verified by unit tests |
| Real end-to-end pilot | Corrected code passed on two clean and two paired corrupted images with random weights | Pass with published trained checkpoints: 16 clean and 16 paired noisy images, all four conditions |

Run the fixture checks from the repository root:

```powershell
python -m pytest tests/test_ttt_evaluation.py -q
```

## Needed for the real pilots

- Nguyen: the corrected review copy is ready to adopt. The original executed
  notebook reports a completed run; its trained weights were not available
  for independent replay. A trained-weight replay is separate from this pilot.
- Tuong: no files needed for the pilot. Both trained checkpoints, the split,
  settings, and adapter are published at revision `339978d`.
- Both: confirm which parameters are allowed to update, optimizer initialization,
  adaptation steps, and prediction timing. Review adapters for label access and
  verify heads stay frozen before accepting pilot results.

Fixture data and toy model outcomes are only evidence for the evaluation
contract. They are not experiment accuracy or evidence of real TTT gains.

## Real CIFAR-100 data pilot

Run `python -m src.check_ttt_cifar100_data` from the repository root with
torchvision's extracted CIFAR-100 archive under `data/raw`. The command saves
or verifies `results/ttt_cifar100_data_pilot.json`.

Passed on real data: 20 development and 20 evaluation images (one per class
0-19 in each list), zero shared source IDs, 20 clean/noisy pairs in identical
order, and exact noise reproduction when generated in reverse order. The
pilot used seed 42 and Gaussian standard deviation 0.1 on pixels in [0, 1].
Both lists are drawn from official test rows for this data-only check. They
are pilot lists, not approved experiment splits or data for tuning the final
model. No model or adapter was run.

## CIFAR-10-C data pilot

Status: passed on November 11, 2026. All 10,000 labels matched in official row
order for Gaussian noise severity 1. Ten clean/corrupted image pairs passed
the input checks, with zero source-ID overlap against ten development images.
The split and evaluation order are saved in
`results/ttt_cifar10c_data_pilot.json`. No model or adapter was run.

Place the official CIFAR-10 Python test batch under
`data/raw/cifar-10-batches-py/test_batch` and the official CIFAR-10-C
`labels.npy` plus the chosen `<corruption>.npy` under `data/raw/CIFAR-10-C/`.
Confirm the corruption name and severity (1–5) before using these inputs for
the experiment. The pilot can run without the model or adapter.

Use the same ordered source IDs for both streams. For zero-based clean test
row `r`, the matching row in the corruption array is
`(severity - 1) * 10000 + r`. Check that the selected label block agrees with
the official clean labels in order and that both streams produce RGB
32-by-32 tensors for the same IDs. The label comparison is an offline
alignment check; no labels go to adaptation. Matching labels alone cannot
prove image identity, so use the original archives and preserve source order.

For the temporary Gaussian-noise severity-1 pilot, use the official
[CIFAR-10-C archive](https://zenodo.org/records/2535967). Extract only
`CIFAR-10-C/labels.npy` and `CIFAR-10-C/gaussian_noise.npy` under
`data/raw`. After the clean CIFAR-10 test batch and both arrays are ready:

```powershell
python -m src.check_ttt_cifar10c_data
```

The checker compares the complete clean and corrupted label order and checks
ten pairs, one per class, without running a model. It writes a pilot manifest
only if the checks pass. Severity 1 is temporary, not the team's final choice.

## Independent checks of the leads' published code

Experiment A: [Nguyen's Kaggle notebook](https://www.kaggle.com/code/vunguyenledinh/notebook9e2b34c05b),
version 1. Save its JSON export locally at `data/raw/nguyen-ttt-review.ipynb`.
The checker pins the reviewed source hash and loads only the definitions it
needs; it does not execute the notebook's download or training cells.

```powershell
python -m src.check_ttt_nguyen_notebook data/raw/nguyen-ttt-review.ipynb
```

Verified on rows 0 and 2 from the notebook's seeded evaluation split.
Frozen inference left the base state unchanged. On clean and severity-5
images, episodic started both images from the same shared weights; online
entered the second image with the first image's updates. Independent reruns
started from the same weights and returned the same predictions. The
checkpoint can be supplied with `--checkpoint <joint_best.pt>`; this local
run used random initialization.

Before full runs:

1. Validate that `order` is a permutation of all evaluation indices.
   The reviewed code accepted `[0, 0]` for two images, leaving the second
   prediction at `-1`.
2. Move running classification accuracy outside `run_ttt`. Its optional
   `y` parameter is used for logging, not gradients, but class labels should
   stay in scoring. Use the same seeded traversal for matched conditions,
   or document any intentional difference.

Experiment B: [Tuong's published revision](https://github.com/dong-quan-tran/acm-online-learning-human-way/tree/317aea7).
Checked the published split files and configuration, including official test
membership for classes 0-19. On real rows `test:6` and `test:7`, each of the
four conditions recorded exactly one prediction per row in order on both
streams. Frozen and standard runs ended at the starting state; online ended
with updated weights. Update counts were 0 for frozen and 2 for adaptive runs.
The published `tests/test_ttt_exp_b.py` suite also passed locally (39 tests),
including the between-image reset and stateless optimizer checks. The
published 128-image smoke test passed on CPU for each clean/noisy condition
(0 updates for frozen, 128 for adaptive).

To reproduce B, use revision `317aea7` in a separate checkout, make the
cached CIFAR-100 data available under its `data/raw`, then run:

```powershell
python -m pytest tests/test_ttt_exp_b.py -q
python -m src.ttt_cifar100 --smoke-test --device cpu --output results/khoi-smoke
```

These initial smoke runs used randomly initialized models. Their accuracy
values are not experiment findings.

### Experiment B: trained checkpoint follow-up

[Revision `339978d`](https://github.com/dong-quan-tran/acm-online-learning-human-way/tree/339978d)
adds the trained supervised and joint seed-42 checkpoints and the result CSVs.
The evaluation code and configuration are unchanged from `317aea7`.

The independent CPU check loaded both checkpoints and ran every condition on
the first 16 official evaluation rows and their paired noisy versions. All
eight runs returned exactly one prediction per image in the saved order.
Frozen runs made zero updates. Standard TTT made 16 updates and restored the
initial state; online TTT made 16 updates and retained changes only in the
shared features. Each condition started from a fresh copy of its checkpoint.

The published CSV integrity checks passed: 2,000 unique ordered evaluation
IDs, all eight summary rows consistent with the 80 saved windows, and update
counts of 0 for frozen and 2,000 for adaptive runs. CSV consistency does not
by itself reproduce the published accuracy.

### Final retest and delivery status

**Experiment A.** The [executed notebook at f2aa445](https://github.com/dong-quan-tran/acm-online-learning-human-way/blob/f2aa445/notebooks/nguyen-expa.ipynb)
reports 8,000 evaluation images per condition. Both original findings were
reproduced. The corrected copy at `notebooks/nguyen-expa-evaluation-review.ipynb`
checks that stream order is a permutation, rejects missing predictions, and
keeps class labels out of the adaptation runner. The two adaptive conditions
use the same seeded order in final evaluation. The original run and its
outputs were preserved; the corrected copy has no saved execution outputs.

```powershell
python -m src.check_ttt_nguyen_notebook notebooks/nguyen-expa-evaluation-review.ipynb
```

Passed on two clean and two paired severity-5 images: one prediction per
image, frozen model unchanged, episodic reset, online carry-forward, and
independent rerun. Repeated, missing, out-of-range and non-integer indices
were rejected. The adapter no longer has a class-label parameter. This check
used random weights; it does not independently reproduce Nguyen's trained
accuracy. Adopting the corrected copy upstream and replaying Nguyen's trained
run are outside this completed local pilot; the original public code has
not been changed.

To reproduce A membership, use NumPy RandomState(0), generate the 50,000-row
training permutation first and then the 10,000-row official test permutation.
Sort the first 5,000 training rows for source development and the remaining
rows for source training. Sort the first 2,000 test rows for TTT development
and the remaining 8,000 for evaluation. The final adaptive traversal is
RandomState(1).permutation(8000) over the sorted evaluation list. Use official
Gaussian noise severity 5 with pixels divided by 255 and the notebook's
model normalization. Smoke source IDs are test:0 and test:2.

**Experiment B.** All four conditions passed the 16-image trained-checkpoint
smoke test on both streams. The full independent CPU replay also completed:
all eight conditions returned 2,000 predictions in the saved order, with
zero updates in frozen runs and 2,000 in adaptive runs. Six accuracies
matched the published results exactly. Clean online TTT was 61.95% locally
vs. 62.10% published; noisy online TTT was 48.80% vs. 48.95%. Both gaps are
-0.15 percentage points. The local run used CPU and the published run used
MPS; the cause of the differences was not isolated. Exact online accuracy
reproduction did not pass, but the evaluation behavior checks passed.

Saved evidence: `results/ttt_exp_b_independent_replay.csv`. Use the membership
and configuration at revision `339978d` for the 2,000-image replay; for each
clean/noisy condition load a fresh checkpoint, use batch size 1 and no shuffle,
and call the published `evaluate_condition` on CPU. Use the supervised
checkpoint only for supervised frozen and the joint checkpoint for the other
three conditions.

**Submission:** This checklist and unresolved-issues record, the data and
evaluation scripts in `src/`, the corrected notebook, the two saved pilot
manifests, and the trained Experiment B replay CSV. No teammate files are
needed to submit the smoke-check assignment.
