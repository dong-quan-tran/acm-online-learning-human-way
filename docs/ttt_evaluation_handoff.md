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

## What I found

### Tuong: CIFAR-100

I loaded Tuong's two trained checkpoints and tried all four methods on 16
clean images and their noisy copies. Each method made one prediction per
image, in the saved order. The frozen models stayed the same, standard TTT
reset after each image, and online TTT kept its changes in the shared layers.
Tuong's 39 tests and his 128-image smoke test also passed locally.

I then ran all eight method and input combinations on the full 2,000-image
test set. The frozen runs made no updates, and the adaptive runs made 2,000
each. The saved CSV totals agreed with all 80 evaluation windows.

Six accuracies matched exactly. Online TTT was the only exception: I got
61.95% on clean images and 48.80% on noisy images, compared with Tuong's
62.10% and 48.95%. That's three fewer correct predictions out of 2,000 on
each stream. I ran on CPU and Tuong ran on MPS, but I haven't confirmed
what caused the difference. The evaluation checks still passed.

### Nguyen: CIFAR-10-C

[Nguyen's executed notebook](https://github.com/dong-quan-tran/acm-online-learning-human-way/blob/f2aa445/notebooks/nguyen-expa.ipynb)
reports a full run on 8,000 evaluation images per condition. I checked that
those rows didn't overlap his 2,000 development rows, and that the severity-5
labels lined up with the clean test labels.

I found two things in his original code:

- With two images, an order of `[0, 0]` tests the first image twice and never
  gives the second one a prediction. This happened in my deliberate bad-order
  test. It doesn't show that his published run skipped any images.
- `run_ttt` takes the real labels to print running accuracy. The update uses
  rotation labels generated from the images, so I didn't find class labels
  being used for updates. I moved accuracy scoring outside the adaptation
  function in the review copy.

The [current review copy](../notebooks/nguyen-expa-evaluation-review.ipynb)
uses the published fixed settings, rejects bad orders, and checks for
missing predictions. The smoke test uses two clean images and their two
noisy copies with untrained weights to check reset and carry-forward.
Independent trained accuracy is still unverified. See the
[current held-out report](ttt_held_out_split_plan.md) for the fixed split,
reported scores, and earlier tuning exposure.

## How to repeat the checks

For Nguyen's review copy, with the public datasets under `data/raw`:

```powershell
python -m src.check_ttt_nguyen_notebook notebooks/nguyen-expa-evaluation-review.ipynb
```

The script also accepts the reviewed version 1 Kaggle JSON export to reproduce
the original findings. It only loads the definitions it needs and skips
training. To test trained weights, supply `--checkpoint <joint_best.pt>`.

For Nguyen's split, use NumPy `RandomState(0)` to permute the 50,000 training
rows first, then the 10,000 official test rows. Sort each slice separately:
the first 5,000 training rows are source development, the remaining 45,000
are source training, the first 2,000 test rows are the fixed evaluation set,
and the remaining 8,000 are the adaptation pool. The online adaptation order is
`RandomState(1).permutation(8000)` over the sorted evaluation rows. Use
official Gaussian noise severity 5, divide pixels by 255, then apply the
notebook's model normalization. The smoke test uses `test:0` and `test:2`.

For Tuong, use [revision 339978d](https://github.com/dong-quan-tran/acm-online-learning-human-way/tree/339978d)
in a separate checkout with the CIFAR-100 cache available under `data/raw`:

```powershell
python -m pytest tests/test_ttt_exp_b.py -q
python -m src.ttt_cifar100 --smoke-test --device cpu --output results/khoi-smoke
```

The smoke command uses random weights; it checks that the code runs. For the
full replay, use the saved membership and config, batch size 1, and no
shuffle. Start each call to `evaluate_condition` from a fresh checkpoint:
the supervised checkpoint for supervised frozen and the joint checkpoint
for the other three methods. The [replay CSV](../results/ttt_exp_b_independent_replay.csv)
has all eight results.

The submission includes the checklist above, the checking scripts, both
pilot image lists, Nguyen's review copy, and Tuong's replay CSV. The two
follow-ups are applying the fixes to Nguyen's original notebook and checking
his trained accuracy if the saved weights become available.
