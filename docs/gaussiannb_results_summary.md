
# Gaussian Naive Bayes — Online Continual Learning

## 1. Objective

Evaluate Gaussian Naive Bayes (GaussianNB) as an online learning baseline on the same CIFAR-100 subset used by the team's other models.

The goal is to compare validation accuracy and training time against logistic regression and SmallResNet under a shared data split and stream configuration.

## 2. Experiment Setup

- Dataset: CIFAR-100, restricted to 20 classes (labels 0–19)
- Training examples: 9,000
- Validation examples: 1,000
- Data loader: Khoi's shared online loader
- Split membership: Original 20-class experiment JSON from VN.LD
- Model: scikit-learn GaussianNB
- Training method: `partial_fit()`
- Training passes: 1
- Batch size: 32
- Model seed: 42
- Stream seed: 42
- Input features: Flattened RGB images (3,072 features)
- Pixel scaling: Preserved from the shared loader
- Device: CPU

The same GaussianNB model is updated sequentially across batches. The complete list of 20 classes is supplied during the first update.

## 3. Evaluation Protocol

Validation accuracy is measured after these numbers of training examples:

1024, 2016, 3008, 4000, 5024, 6016, 7008, 8000, 9000

These checkpoints correspond to the recorded evaluation points in the team's previous 20-class experiment.

No zero-example accuracy is reported because GaussianNB requires an initial training update before prediction.

Training time is measured cumulatively around `partial_fit()` calls, excluding validation time and data loading.

## 4. Reproduction

Run from the repository root after installing the project's dependencies and obtaining the shared membership JSON.

```bash
python src/train_gaussiannb.py --membership path/to/shared_c20_split.json --batch-size 32 --seed 42
```

The command writes:

`results/gaussiannb_results.csv`

To generate the learning curve:

```bash
RESULTS_CSV=results/gaussiannb_results.csv python src/plot_results.py
```

The plotting script writes:

`results/figures/online_learning_curve.png`

## 5. Tests

The GaussianNB implementation has five passing unit tests.

The plotting utilities have six passing tests.

A full end-to-end training run using the original shared membership file is still pending.

## 6. Experimental Results

Pending execution with the shared 20-class membership JSON.

- Final validation accuracy: TBD
- Total model-update training time: TBD
- Number of training updates: TBD
- Numerical warnings: TBD

## 7. Comparison With Other Models

The team's previous logistic regression and SmallResNet results will be used for comparison after the GaussianNB experiment completes.

Comparison criteria:

- Validation accuracy at shared checkpoints
- Final validation accuracy
- Model-update training time
- Learning-curve behavior
- Computational cost and model limitations

## 8. Limitations

GaussianNB assumes features are conditionally independent given the class. This assumption is restrictive for image data because neighboring pixels are often correlated.

GaussianNB also models each feature using a class-conditional Gaussian distribution, which may not accurately represent all pixel-value distributions.

These limitations will be considered when interpreting the experimental results.

## 9. Status

Implementation and preliminary unit tests completed.

Awaiting the original shared membership JSON for full experimental execution and final comparison.
