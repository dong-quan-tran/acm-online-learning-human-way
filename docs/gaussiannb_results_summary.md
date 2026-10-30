# Gaussian Naive Bayes — Online Continual Learning

## 1. Objective

Evaluate Gaussian Naive Bayes (GaussianNB) as an online learning baseline on the CIFAR-100 dataset and compare its accuracy and training time with the team's other models.

## 2. Experiment Setup

- **Dataset:** CIFAR-100, 20 classes (labels 0–19)
- **Training examples:** 9,000
- **Validation examples:** 1,000
- **Data loader:** Khoi's shared online loader
- **Data split:** VN.LD's original 20-class split
- **Model:** scikit-learn GaussianNB
- **Training method:** `partial_fit()`
- **Training passes:** 1
- **Batch size:** 32
- **Model and stream seed:** 42
- **Input:** Flattened RGB images (3,072 features), scaled to [0, 1]
- **Device:** CPU

The model was updated incrementally using batches of 32 images. All 20 classes were specified during the first update.

The original training stream order was verified against VN.LD's saved split.

## 3. Evaluation Protocol

Validation accuracy was measured after processing:

1,024, 2,016, 3,008, 4,000, 5,024, 6,016, 7,008, 8,000, and 9,000 examples.

Training time was measured cumulatively around `partial_fit()` calls, excluding data loading and validation.

## 4. Experimental Results

| Examples Processed | Validation Accuracy |
|---|---|
| 1,024 | 22.8% |
| 2,016 | 22.1% |
| 3,008 | 22.1% |
| 4,000 | 20.7% |
| 5,024 | 20.8% |
| 6,016 | 21.6% |
| 7,008 | 20.6% |
| 8,000 | 20.7% |
| 9,000 | 20.6% |

**Final results:**
- **Final validation accuracy:** 20.6%
- **Total model-update time:** 0.788 seconds
- **Training updates:** 282
- **Training examples processed:** 9,000
- **Device:** CPU
- **Numerical training warnings:** None reported

Validation accuracy remained approximately between 20% and 23%, without sustained improvement as additional examples were processed.

## 5. Reproduction

From the repository root, install the required dependencies and run:

```bash
pip install -r requirements.txt
python src/train_gaussiannb.py --membership data/shared_c20_membership.json --batch-size 32 --seed 42
```

Results are saved to:

`results/gaussiannb_results.csv`

Generate the learning curve:

```bash
RESULTS_CSV=results/gaussiannb_results.csv FIGURE_NAME=gaussiannb_learning_curve.png python src/plot_results.py
```

The graph is saved to:

`results/figures/gaussiannb_learning_curve.png`

## 6. Final Verification

The following tests were run in Google Colab using Python 3.13.18.

### GaussianNB Model Tests

Command:
`python -m pytest tests/test_gaussiannb.py -v`

Result:
`5 passed in 12.47s`

### Plotting Tests

Command:
`python -m pytest tests/test_plot_results.py -v`

Result:
`6 passed in 1.74s`

### Graph Generation

Environment variables:
- `RESULTS_CSV=results/gaussiannb_results.csv`
- `FIGURE_NAME=gaussiannb_learning_curve.png`

Command:
`python src/plot_results.py`

Output:
`Learning curve saved to: /content/acm-online-learning-human-way/results/figures/gaussiannb_learning_curve.png`

The generated graph file was confirmed to exist.

**Total: 11 tests passed, 0 failed.**
### Training Stream Verification

To verify consistency with VN.LD's experiment, I compared the GaussianNB loader's training indices against VN.LD's saved stream file (`online_c20_m42_s42_20261028-173041_split.json`).

Both contained exactly 9,000 training samples. The comparison confirmed that all sample IDs were identical and appeared in exactly the same order.

**Verification output:**

```text
Nguyen's samples: 9000
GaussianNB samples: 9000
Same sample IDs: True
Exact training order matches: True
All 9,000 verified: True
```

This confirms that GaussianNB used the same training sample sequence as VN.LD's saved stream, ruling out differences in training sample selection and order as explanations for the observed accuracy difference.
## 7. Comparison with ResNet

I compared my GaussianNB results with VN.LD's ResNet experiment. Both models used the same 9,000 training samples in the same order, with 282 updates.

| Metric | GaussianNB | ResNet |
|---|---|---|
| Final accuracy | 20.6% | 43.2% |
| Training time | 0.788s | 4.651s |
| Device | CPU | Tesla T4 GPU |

### Observations

GaussianNB started with higher accuracy (22.8% vs. 15.6% at 1,024 samples), but stayed around 20–23% throughout training. ResNet started lower but kept improving, reaching 43.2%.

I think GaussianNB performed better initially because it could learn basic patterns from fewer samples. However, its assumption that pixel features are independent may limit its performance on images. ResNet can learn more complex image features, which could explain why it improved with more data.

### Limitations and Next Steps

This comparison is based on one run per model. Also, training times aren't directly comparable because the models ran on different hardware and used different training methods.

As a next step, I would test both models with multiple random seeds to see whether the accuracy difference stays consistent.
### Reproduce Comparison Graph

The combined results are saved in `results/gaussiannb_vs_resnet_results.csv`.

To regenerate the comparison graph, run:

```bash
RESULTS_CSV=results/gaussiannb_vs_resnet_results.csv FIGURE_NAME=gaussiannb_vs_resnet.png python src/plot_results.py
```

Output: `results/figures/gaussiannb_vs_resnet.png`

