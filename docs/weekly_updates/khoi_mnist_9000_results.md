# Khoi: required MNIST 9,000/1,000 experiment

Completed from scratch with all ten digits mixed, one pass, no replay, batch
size 32, split/model/stream seeds 42, SmallResNet (1 input channel, 10 outputs),
Adam 0.001. No official test-set evaluation. Exact selected original source rows
and order are archived; 50,000 remaining training-source rows are unused.

- Training: 9,000; validation: 1,000; no overlap.
- Updates: 282 (281 batches of 32, final batch 8).
- Final validation accuracy: 96.40%.
- Macro precision: 96.5681%.
- Macro recall: 96.2644%.
- Macro F1: 0.963253.
- Training compute: 133.07s (excludes loading/evaluation).
- Timed run wall time: 194.62s (includes loading/evaluation
  during the run, not initial dataset setup).
- Device: CPU; hardware and package versions in run.json.
- Regression suite: 29 passed in 43.63s.

```powershell
python -m src.train_online --config configs/mnist_9000.yaml --output data/processed/mnist_9000_seed42
python -m src.audit_mnist --run data/processed/mnist_9000_seed42 --output results/mnist_9000_seed42/audit
```

The original 54,000/6,000 experiment and its 99.08% score are preserved as a
separate historical run. This run matches the CIFAR example counts but does not
establish model/settings parity. The same-model CIFAR comparison still awaits
Nguyen's original online run artifacts. See notebooks/mnist_9000_comparison.ipynb.

Independent checkpoint audit reproduced all four metrics: 964 correct, 36
errors on 1,000 validation images. Loader memberships match, model state stayed
unchanged, and no official test images were used. The 9,000/1,000 notebook
executed all four code cells successfully and passed notebook validation.
