# TTT Results and Comparison — Experiment B

## Overview

For Experiment B, I analyzed Tuong's results on CIFAR-100 (classes 0–19) to see how standard TTT and online TTT perform on clean images and images with Gaussian noise.

I compared both methods against the joint-frozen baseline because they use the same trained checkpoint. I also looked at runtime and accuracy across different windows to understand how the model behaves during adaptation.

These are preliminary results from seed 42.

## Accuracy Results

| Method | Clean | Gaussian Noise |
|---|---:|---:|
| Joint Frozen | 65.85% | 35.05% |
| Standard TTT | 65.15% | 35.60% |
| Online TTT | 62.10% | 48.95% |

Compared with the joint-frozen baseline:

- Standard TTT: −0.70 pp on clean data, +0.55 pp on noisy data.
- Online TTT: −3.75 pp on clean data, +13.90 pp on noisy data.

The main thing I noticed is that **online TTT helped a lot with Gaussian noise but actually performed worse on clean images**.

Standard TTT barely improved noisy accuracy. This suggests that carrying updates from previous test samples may be useful when the model keeps seeing the same type of corruption.

## Windowed Accuracy

I also plotted accuracy across 10 windows, each containing 200 images.

### Clean Images

![Clean accuracy](../results/exp_b_seed42_clean.png)

Online TTT stayed below the frozen baseline for almost the entire test stream. This shows that adaptation is not always beneficial, especially when the original model already performs reasonably well.

### Gaussian Noise

![Gaussian noise accuracy](../results/exp_b_seed42_gaussian_noise.png)

Online TTT started at 36.5% accuracy and reached 55% by window 4. It performed better than the frozen baseline in 9 out of 10 windows.

One possible explanation is that the model gradually adjusts to the noisy distribution because online TTT keeps its updates instead of resetting after every sample.

## Runtime Comparison

| Method | Clean | Gaussian Noise |
|---|---:|---:|
| Joint Frozen | 7.09 s | 6.67 s |
| Standard TTT | 34.08 s | 38.81 s |
| Online TTT | 38.67 s | 38.27 s |

Both TTT methods performed 2,000 adaptation updates per run.

Online TTT took roughly 5–6 times longer than the frozen baseline. So even though it improved noisy accuracy, that improvement comes with additional computation.

## My Takeaways

From these results, I think online TTT looks more promising than standard TTT for this particular corruption setting. However, the drop in clean accuracy is something we should pay attention to.

I also don't think we can conclude that online TTT is generally better yet. We only have one seed and one corruption type, so the results might change with other settings.

## Next Steps

- Add seeds 43 and 44 to check whether the improvement is consistent.
- Calculate mean and standard deviation across seeds.
- Include Nguyen's Experiment A results.
- Compare the benefits of adaptation against its runtime cost.

**Overall:** Online TTT showed a strong improvement under Gaussian noise (+13.90 percentage points), but it also reduced clean accuracy and required much more computation. The next step is to see whether this pattern holds across multiple seeds.
