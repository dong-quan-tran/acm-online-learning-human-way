# Tuong: Rotation-Based Test-Time Training

Source: Jason-s-report.pptx, supplied by Dong Quan and identified as Tuong's presentation.
Type: Assistant-prepared summary, not a verbatim transcript.

## Submitted Work

The presentation reviews Sun et al.'s original TTT paper. Source training combines classification and four-way rotation prediction. At test time, the proposed method updates the shared feature extractor through rotation loss while keeping both heads frozen.

Standard TTT resets after each image. Online TTT keeps updates for the next image. The presentation discusses weak rotation signals, conflicting task updates, drift, normalization, and extra computation.

## Proposed Experiment B

- Use CIFAR-100 classes 0-19 and the existing unmasked 100-output head.
- Train matched source models, using GroupNorm in the proposed TTT setup.
- Compare supervised frozen, joint frozen, standard TTT, and online TTT.
- Use clean images and one fixed Gaussian-noise condition.
- Keep labels outside adaptation and report accuracy, harm, runtime, updates, and three-seed variation.

## Status

Dong Quan confirms presentation and advisor approval. The presentation explicitly says the proposed experiment has not been run. It reviews one paper; the originally requested second-paper comparison is not shown.

Paper: https://proceedings.mlr.press/v119/sun20b/sun20b.pdf
