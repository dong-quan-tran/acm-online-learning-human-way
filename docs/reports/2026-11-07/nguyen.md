# Nguyen: Test-Time Training

Source: Nguyen-s-report.pptx, supplied by Dong Quan.
Type: Assistant-prepared summary, not a verbatim transcript.

## Submitted Work

The slides explain distribution shift and compare predefined training transformations with learning from test images. They introduce self-supervision and raise questions about measuring shift and choosing an auxiliary task. They also discuss a teacher/student self-supervision example. Background dates and dataset-size claims are not independently verified here.

## Proposed Experiment A

1. Train on CIFAR-10 using classification and rotation losses.
2. Evaluate on CIFAR-10-C.
3. Update the shared backbone using rotation loss without test class labels.
4. Predict the image class using the updated backbone.

The proposal does not yet fix all baseline, reset, or tuning settings. The current team assignment adds those implementation requirements.

## Status

Dong Quan confirms that Nguyen presented to Dr. Athitsos and received approval to proceed. The uploaded slides contain a proposal, not runnable adaptation code or measured team TTT results.

Reading identified in the slides: Y. Sun's 2023 UC Berkeley dissertation on test-time training.
