# Alka: Entropy-Based Test-Time Adaptation

Source: Alka-s-report.docx, supplied by Dong Quan.
Type: Assistant-prepared summary, not a verbatim conversion.

## Submitted Work

The report reviews Tent and EATA and compares their preparation, updates, benefits, and limits.

Tent reduces prediction entropy by updating normalization scale and shift while using incoming batch statistics. EATA filters unreliable or redundant samples and adds Fisher-based protection against forgetting. The report notes that confidence can increase while predictions remain wrong.

## Project Recommendation

Compare frozen, normalization-only, and Tent conditions using the same checkpoint, images, order, and batch size. Keep labels outside adaptation. Record accuracy, entropy, runtime, and update counts, and check clean-image harm and reset rules.

This was Alka's submitted recommendation. The current team assignment instead starts the two approved rotation-based TTT experiments; Tent/EATA remain possible later comparisons.

## Status

The two-paper report and comparison were received. Its numerical examples come from the papers, not team adaptation runs. Separate presentation slides were not among the four uploaded files indexed here.

Papers:
- https://arxiv.org/pdf/2006.10726
- https://proceedings.mlr.press/v162/niu22a/niu22a.pdf
