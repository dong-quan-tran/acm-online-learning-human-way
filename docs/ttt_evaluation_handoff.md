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
| Source-ID overlap / saved order | Real data pilot passed; final split pending | Real data pilot passed; final split pending |
| Corrupted input preparation | Official Gaussian noise severity 1 pilot passed; final settings pending | Real repeatable noise pilot passed; final strength pending |
| Label-free callback and one prediction per image | Fixture passed; actual adapter pending | Fixture passed; actual adapter pending |
| Frozen invariance; standard reset; online carry | Fixture passed; actual checkpoint pending | Fixture passed; actual checkpoint pending |
| Independent model and optimizer state | Fixture passed; actual checkpoint pending | Fixture passed; actual checkpoint pending |
| Real end-to-end pilot | Pending | Pending |

Run the fixture checks from the repository root:

```powershell
python -m pytest tests/test_ttt_evaluation.py -q
```

## Needed for the real pilots

- Nguyen: agreed development/evaluation source IDs, official corruption file
  and severity, preprocessing, joint checkpoint, adapter, and reset mode.
- Tuong: agreed twenty-class development/evaluation IDs, Gaussian noise
  strength and seed, preprocessing, joint checkpoint and adapter.
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
