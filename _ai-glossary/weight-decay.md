---
title: Weight Decay
related:
  - Regularization
  - Adam
  - Overfitting
---
**Weight decay** penalizes large weights (L2-style), encouraging simpler solutions. In AdamW it is applied separately from the adaptive gradient update.

It is a standard stabilizer for transformer training; tune jointly with learning rate.
