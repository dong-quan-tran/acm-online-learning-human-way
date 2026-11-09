# Khoi: Evaluation Pitfalls in Test-Time Adaptation

Source: Khoi-s-report.docx, supplied by Dong Quan.
Type: Assistant-prepared summary, not a verbatim conversion.

## Submitted Work

The report reviews On Pitfalls of Test-Time Adaptation and its TTAB benchmark. It explains how learning rates, adaptation steps, source models, reset rules, and distribution shifts can change comparisons.

It distinguishes episodic resets from online carryover. Earlier updates can affect later batches, and more adaptation can hurt accuracy. Restoring parameters and Fisher regularization are discussed as safeguards that still depend on settings.

## Project Recommendation

- Agree on evaluation rules before comparing methods.
- Compare against a frozen model.
- Track performance throughout the stream, not only the final score.
- Match splits, seeds, batch sizes, stream order, and update rules.

## Status

The report was received. It reviews one paper and provides evaluation recommendations, not completed team adaptation results. A second-paper comparison, separate slides, and a full project checklist were not demonstrated in this uploaded document.

Paper: https://arxiv.org/abs/2306.03536
