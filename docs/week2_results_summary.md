# Week 2 Results Summary

## Overview
This week, we shifted from Task 1/Task 2 forgetting experiments to shuffled online learning, tracking accuracy as training examples are processed.

## My Contributions
- Updated the CSV template and plotting script for the new experiment.
- Changed the visualization to accuracy vs. training examples processed.
- Tested the script in Google Colab using synthetic data.
- Added six automated tests covering valid, incomplete, invalid, missing, and empty CSV data.
- All six tests passed successfully (evidence attached to PR #6).

## Setup and Testing
Run from the repository root:

Install dependency:
```bash
python -m pip install matplotlib
```

Generate plot:
```bash
python src/plot_results.py
```

Run tests:
```bash
python -m unittest discover -s tests -p "test_plot_results.py" -v
```

Plot output: `results/figures/online_learning_curve.png`

## Results
The plotting script was successfully tested in Google Colab using both synthetic data and Nguyen's actual CIFAR-100 experiment results. All six automated tests passed. The real learning curve shows a final validation accuracy of 39.5% after processing 9,000 training examples.
## Next Steps
- Test the plotter with Nguyen's real CSV without modifying its columns.
- Generate the final learning curve with run ID and evaluation split.
- Record actual settings, sample counts, final accuracy, training time (if available), and limitations.
