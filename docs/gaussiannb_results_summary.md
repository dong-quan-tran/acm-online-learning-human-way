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
