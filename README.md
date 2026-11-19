# Fixed held-out TTT: Alka's logging and plotting handoff

## Files
- `heldout_results_template.csv`: agreed-format **proposal**; leads should confirm field definitions before pilots.
- `plot_heldout_ttt.py`: reusable plotting and summary script.
- `SYNTHETIC_example.csv`: **FAKE data only**, to test the script; never report as experimental results.

## CSV conventions
- `experiment`: `A` or `B`; `condition`: free-text run label (e.g. `pilot`).
- `adaptation_stream_condition`: `clean` or `noisy` (independent runs from fresh checkpoints).
- `heldout_condition`: `clean` or `noisy` (same fixed held-out image IDs within each experiment).
- `heldout_accuracy`: fraction 0–1, **not percent**; equal to `correct_prediction_count / heldout_image_count`.
- `adaptation_images_processed`: cumulative image count, including **0** before TTT.
- `adaptation_update_count`: cumulative optimizer steps since start of adaptation.
- `adaptation_time_seconds`: cumulative adaptation-only elapsed time since start; excludes evaluation.
- `heldout_evaluation_time_seconds`: time for **this** held-out evaluation, not cumulative.
- `source_checkpoint`, `code_version`, `config_path`, `split_file_path`: reproducibility identifiers/paths.
- `seed`: random seed for this run. For each seed/stream, log both held-out conditions at every checkpoint.
- Reference accuracies should be evaluated on the **same fixed held-out set**; do not reuse old frozen accuracy measured on a different set.

## Run
```bash
pip install pandas matplotlib
python plot_heldout_ttt.py SYNTHETIC_example.csv --out demo_plots
python plot_heldout_ttt.py results_a.csv results_b.csv --reference-csv frozen_references.csv --out real_plots
```
Optional `frozen_references.csv` schema:
`experiment,seed,heldout_condition,method,accuracy`
with `method` = `joint_frozen` or `supervised_frozen` and accuracy as fraction 0–1.

Outputs: **separate plots for A/B and clean/noisy adaptation streams**, each with clean/noisy held-out panels; plus `accuracy_changes.csv` (percentage-point changes relative to online starting accuracy and, when reference CSV provided, joint frozen).

## Pilot checks to coordinate with Khoi
At least 0, 100, 200 image evaluation points; no held-out/adaptation image overlap; no model/optimizer/RNG changes during evaluation; one prediction per held-out image; held-out labels used only for scoring; evaluation must not change subsequent adaptation. These checks are **not** implemented by this plotting script and require experiment-side verification.

## Interpretation
Report separate clean/noisy adaptation runs, per-seed curves, frozen baselines, accuracy gains/losses, adaptation and evaluation times. Do not claim these synthetic results are real or mix this new held-out protocol with earlier stream-window accuracies.
