---
title: Early Stopping
related:
  - Overfitting
  - Evaluation
  - Fine-Tuning
---
**Early stopping** ends training when a validation metric stops improving for a patience window, then reloads the best checkpoint. It is a simple, strong regularizer for [fine-tuning](/ai-glossary/fine-tuning/) when you have a trustworthy validation set.

### How to run it
1. Hold out a fixed validation split that matches production distribution
2. Evaluate every $N$ steps (not only per epoch)
3. Stop after $P$ evals with no improvement; keep the best weights

### Pitfalls
- Noisy metrics need larger patience or smoothed scores
- Optimizing only train loss invites [overfitting](/ai-glossary/overfitting/) even if early stopping is “on”
- For domain [CPT](/ai-glossary/continued-pretraining/), track both LM loss and task probes—stop on the metric you care about shipping

Early stopping complements [dropout](/ai-glossary/dropout/), weight decay, and smaller learning rates; it does not replace good data hygiene.
