# Alka PR #9 review and preservation notes

## Original Week 2 work (PR #8)

PR #8 supplied the original 39.5% CIFAR ResNet results, a real-data learning curve, six plotting tests, and a Week 2 summary update. Its CSV has Git blob SHA 9d0190c93f0b443a3efc0db190ae425ff293408e, identical to results/smallresnet-cifar-100.csv on main at review time. The six plotting tests are preserved in PR #9 (blob 4d1ac7cd53e64bf58ce4d8b46a893bf105ef320e). The original curve can be regenerated; do not replace the original run with the newer 43.2% run. This note preserves the Week 2 completion record without carrying forward the old pending-work list.

## Comparison provenance and limitations

The original ResNet run is online_c20_m42_s42_20261015-200612 (39.5%). The updated comparison uses online_c20_m42_s42_20261028-173041 (43.2%), from Nguyen's updated src/rawsgd-cifar-100.ipynb at commit 3d356f9efa7bbcff1c07eb49936a53bf1692a8c0. In the combined CSV, resnet_nguyen_seed42 is an alias for this updated run, not the original run.

ResNet uses random crop/flip augmentation, CIFAR mean/std normalization inside the model, and a 100-output head evaluated without restricting predictions to the twenty selected labels. GaussianNB uses twenty classes and flattened RGB pixels scaled to [0,1], without augmentation. These are model-pipeline comparisons, not a change-only-the-model experiment. GaussianNB timing measures partial_fit updates on CPU; ResNet ran on a Tesla T4 and uses a different timing scope. Do not infer an end-to-end speed ratio. The report's explanations are hypotheses based on one run per model.

## Reproduce the stream comparison

Alka reports that her training sequence matched Nguyen's saved stream. The review has not independently reproduced that report. Obtain Nguyen's original online_c20_m42_s42_20261028-173041_split.json, retain it unchanged, and archive it with his permission in data/original_splits/. Do not fabricate an original artifact from the shared membership.

Run from the repository root:

```bash
python -m src.verify_shared_stream --original data/original_splits/online_c20_m42_s42_20261028-173041_split.json
```

The script checks all training IDs and their order, all validation IDs and their order, counts, uniqueness, and separation. Record actual output after execution. No successful execution of this new script is claimed here.

## Figure and merge checklist

The committed comparison image is currently results/figures/gaussiannb_vs_resnet (1).png; the intended canonical name is results/figures/gaussiannb_vs_resnet.png. Regenerate from the combined CSV with:

```bash
RESULTS_CSV=results/gaussiannb_vs_resnet_results.csv FIGURE_NAME=gaussiannb_vs_resnet.png python src/plot_results.py
```

Commit the canonical image and remove the duplicate only after checking the generated graph. Run GaussianNB, plotting, and loader tests on the final branch and retain output. Adding this note does not certify those tests have run. Merge PR #9 only after reviewing the final changes; then close PR #8 as superseded, retaining both branches. The three-model comparison still needs Logistic Regression added from its exported results.
