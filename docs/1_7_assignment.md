# Next Week’s Tasks — November 1–7
## Test-Time Learning from Unlabeled Data

Following our meeting with Dr. Athitsos, we will explore how a model can keep learning from new images without using their class labels.

This week will focus on **preparation**: three focused literature reviews and one small trial. Everyone should work independently rather than wait for another member’s submission.

## Shared Expectations

- Literature reviewers: review two papers in depth; a third is optional.
- Explain the methods in your own words—not just a list of techniques.
- Include a comparison table, limitations, and your recommendation for our project.
- Keep reports and slides self-contained so you can present without switching between papers or notebook cells.
- Upload reports, slides, code, and results to your own GitHub branch. Include paper links and clear file names.
- Finish any small remaining cleanup from your previous assignment without overwriting existing results.

## Tuong — Self-Supervised Test-Time Training

### Tasks
Start with *Test-Time Training with Self-Supervision for Generalization under Distribution Shifts* and choose one closely related paper.

Explain:
- What self-supervised task provides the learning signal.
- What source training is needed before deployment.
- Which parts of the model change at test time.
- How the method learns without class labels.
- What assumptions or limitations could prevent it from working in our setting.

### Submit
- Short report and presentation slides.
- Two-paper comparison table.
- A recommendation explaining what we would need to change in our current pipeline.

Pause the earlier SGD–AdaGrad proposal while we investigate this direction.

## Alka — Entropy-Based Test-Time Adaptation

### Tasks
Start with *Tent: Fully Test-Time Adaptation by Entropy Minimization* and choose one related paper that improves the method or examines its limitations.

Explain:
- What prediction entropy means and why it is used as a learning objective.
- Which parameters and normalization statistics change.
- What preparation the method requires.
- Why a model becoming more confident does not necessarily mean it becomes more accurate.
- What safeguards or baselines we should use.

### Submit
- Short report and presentation slides.
- Comparison table covering both methods.
- A recommendation for a practical first adaptation method and its main risks.

## Khoi — Evaluation Protocols and Failure Cases

### Tasks
Choose two papers focused on reliable test-time adaptation, evaluation, or failure under changing data.

Investigate:
- Prediction before versus after adaptation.
- Resetting the model versus carrying updates through a stream.
- Effects of batch size, image order, and changing distributions.
- How to avoid label leakage and tuning on final evaluation data.
- How papers detect and report cases where adaptation hurts performance.

### Submit
- Short report and presentation slides.
- Comparison table of the papers’ evaluation settings.
- A proposed evaluation checklist for our project, including baselines, reset rules, metrics, and label separation.

## Nguyen — Small Feasibility Trial

### Tasks
Use an existing SmallResNet checkpoint and an official or clearly documented minimal Tent implementation.

- Start with CIFAR classes 0–19 and one fixed corruption, such as noise or blur.
- Compare the frozen model with the adapted model.
- Use the same checkpoint, images, order, and batch size for both conditions.
- Keep class labels outside the adaptation code; use them only for evaluation.
- State whether predictions happen before or after adaptation and when the model resets.
- Record the settings, sample count, accuracy, update count, hardware, and runtime.

You do not need to wait for the literature reviews. Use the chosen implementation’s documented protocol and explain any changes.

### Submit
- Runnable notebook or script with setup instructions.
- Small results table and learning curve, if applicable.
- Brief report explaining what worked, what failed, and what we should test next.

This is a feasibility trial, not a full benchmark. If setup fails, document the blocker and what you tried. Do not claim an improvement unless the comparison supports it.

## Literature Review Format

For each paper, cover:

1. Problem and main idea.
2. Source-training and test-time procedures.
3. Available data, labels, and model parameters.
4. Main results and evaluation setup.
5. Limitations and relevance to our project.

End with one concrete experiment recommendation: what we would change, what we would keep fixed, and what result would help answer the question.

## End-of-Week Goal

Bring together:
- Three complementary literature reviews.
- One small trial with practical lessons.
- A recommended first experiment and evaluation protocol.
- Any questions we need Dr. Athitsos to clarify.

The goal is to show that we understand the methods and can design a reliable experiment—not simply produce a higher accuracy score.
