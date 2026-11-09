# Test-Time Training with Self-Supervision

## How TTT works

**Test-Time Training (TTT)** updates a trained model using a test image before predicting its class. The model has a **feature extractor**, a **classification head**, and a **rotation head**. The feature extractor is a sequence of CNN layers that turns an image into numerical features. The classification head uses these features to predict the class. The rotation head uses these features to predict the rotation.

**Source training** happens before deployment. Train all three components using classification loss and rotation loss. Classification loss uses class labels. Rotation loss uses **rotation labels**: 0°, 90°, 180°, or 270°. We know the rotation labels because we apply the rotations ourselves. The feature extractor learns from both losses.

**At test time**, create rotated versions of the test image and calculate rotation loss. In Equation 3 of the paper, update only the feature extractor. Both heads remain **frozen**, meaning their parameters do not change. Backpropagation calculates how changing the feature extractor changes rotation loss, even through the frozen rotation head. Then use the updated feature extractor and frozen classification head to classify the original image. Test-time updates use rotation labels, with class labels kept separately for measuring accuracy.

| Method | Test-time updates | Between test images |
|---|---|---|
| Supervised frozen model | None; source training uses classification only | Keep the source model |
| Frozen joint model | None | Keep the source model |
| Standard TTT | Feature extractor | Reset to the source model |
| Online TTT | Feature extractor | Keep the updated model |

## Evidence and limitations

On CIFAR-10.1, Standard TTT reduced error from 16.7% for the frozen joint model to 15.9%: a reduction of 0.8 percentage points. However, reducing rotation loss does not guarantee better classification. Rotation prediction must help the feature extractor learn features useful for classification. Rotation prediction can be unclear for some images, such as airplanes in the sky. Online TTT also assumes that test images come from the same or a slowly changing distribution. TTT adds computation and does not train the classification head to recognize new classes.

## Changes needed in our pipeline

Our SmallResNet currently trains on class labels in one pass and has no rotation head. Add a rotation head and repeat source training using classification loss and rotation loss. However, a new, untrained rotation head is insufficient. Use **GroupNorm**, which normalizes groups of features within each image, as the paper does for test-time updates using versions of one image. Train the frozen joint model with GroupNorm too.

Keep classes 0-19 and the 100-output classification head. Compare supervised frozen, frozen joint, Standard TTT, and Online TTT. Start frozen joint and both TTT modes from the same jointly trained source checkpoint; use matched test images, image order, and preprocessing. Test one fixed noise condition and clean images. Keep class labels outside test-time updates, choose settings on separate development data, and report accuracy and computation time. Our pipeline has no TTT results yet.

To test reset behavior alone, match adaptation steps and other update settings between standard and online TTT, and state how optimizer state resets. The paper uses ten steps per image for standard TTT and one for online TTT. If using those different budgets, report them clearly rather than treating the comparison as a reset-only test.

## Reference

Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A. A., and Hardt, M. (2020). Test-Time Training with Self-Supervision for Generalization under Distribution Shifts. ICML. https://proceedings.mlr.press/v119/sun20b.html

Editorial note, November 9, 2026: added the reference and supervised-frozen baseline, fixed capitalization, and clarified matched checkpoints and update budgets. The original submission remains in Git history and on Tuong's branch.
