---
title: Overfitting
related:
  - Underfitting
  - Regularization
  - Dropout
  - Early Stopping
  - Evaluation
---
**Overfitting** means the model fits training idiosyncrasies and generalizes poorly. Classic signal: train loss keeps falling while validation / held-out metrics worsen.

### Common causes
- Small or narrow datasets relative to capacity (full fine-tunes on tiny corpora)
- Training too long without [early stopping](/ai-glossary/early-stopping/)
- Weak [regularization](/ai-glossary/regularization/) ([dropout](/ai-glossary/dropout/), weight decay)

### Mitigations
More diverse data, stronger regularization, augmentation, smaller adapters / lower LoRA rank, and honest [evaluation](/ai-glossary/evaluation/) on held-out domains. Opposite failure mode: [underfitting](/ai-glossary/underfitting/) — capacity or training too weak to fit the signal.
