# Held-out TTT handoff (November 14-20)

Both JSON files are drafts until the leads confirm the sizes. The original
source training and development membership stays the same.

| Experiment | Source train | Source development | TTT development | Adaptation | Held out |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (Nguyen) | 45,000 | 5,000 | 2,000 | 6,000 | 2,000 |
| B (Tuong) | 9,000 | 1,000 | Same development set | 1,500 | 500 |

A uses `development_ids` for TTT tuning and `source_development_ids`
for source development. B uses `development_ids` for validation.
Both held-out lists reuse images from earlier evaluations. Keep clean and
noisy views tied to the same original image ID.

## Loading and pilot

1. Confirm sizes with the leads. If sizes change, permute the sorted prior
   evaluation IDs once using NumPy RandomState(42), take the requested
   adaptation prefix in that order, and sort the rest for held-out
   evaluation. Keep the source lists as saved.
2. Nguyen should load `data/ttt_exp_a_held_out_draft.json`. Parse each ID as
   `split:row`, load the corresponding official CIFAR-10 row, and use the
   same test row offset in the severity-5 Gaussian-noise array. Iterate
   adaptation IDs in listed order. Use the held-out list for both views.
3. Tuong should run on his revised branch with the B draft JSON available:

   ```powershell
   python -m src.ttt_held_out --split-file data/ttt_exp_b_held_out_draft.json --max-adapt-images 200
   ```

   Use his published config and checkpoints and Alka's schema map when ready.
4. For each lead, ask for the pilot CSV, ordered prediction evidence, and
   checks JSON or logs. Run clean and noisy adaptation separately from the
   same checkpoint. For each run, score both held-out views at 0, 100 and 200.
   For longer runs, score a final partial group too.

## Pass/fail checklist

| Check | A | B |
| --- | --- | --- |
| All source IDs unique across saved lists | Pass (60,000 IDs) | Pass (12,000 IDs) |
| Adaptation and held-out cover the previous evaluation pool | Pass (8,000) | Pass (2,000) |
| Published split parser accepts draft | A runner integration pending | Pass (1,500 adaptation, 500 held out) |
| Same held-out IDs at every checkpoint, one prediction per image per view | Pilot pending | Pilot pending |
| Checkpoints at 0, 100 and 200; final partial group when applicable | Pilot pending | Pilot pending |
| Evaluation leaves weights, normalization, optimizer and adaptation RNG unchanged | Pilot pending | Pilot pending |
| Adaptation resumes from the same model and optimizer after evaluation | Pilot pending | Pilot pending |

Mark the actual outputs here before signing off on longer runs. The split
checks above are local validation and do not imply that either pilot ran.

## Draft pilot follow-up

Nguyen's published notebook still builds one evaluation list internally.
The helper in `src/ttt_exp_a_split.py` reads the two ordered lists, validates
their row IDs and rejects overlap with development. It returned 6,000
adaptation and 2,000 held-out rows in a local check. Nguyen's runner still
needs to call it, adapt on the first list, and score only the second.

The local B draft pilot uses Tuong's revision `010226f`, CPU, the published
checkpoints, and `--max-adapt-images 205`. Expected checkpoints are 0, 100,
200 and 205 for both held-out views in each adaptive run.

B draft pilot passed. Both online streams scored 500 images per view at
0, 100, 200 and 205. Both frozen baselines scored both views at 0 only.
All four runs reported passing invariance checks, and the CSV had 20 rows.
See [pilot CSV](../results/ttt_exp_b_draft_205.csv) and
[runner checks](../results/ttt_exp_b_draft_205_checks.json). The shared
copies use repo-relative paths and record the actual runner revision,
010226f. The original local output picked up the parent checkout revision.
