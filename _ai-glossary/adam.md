---
title: Adam
synonyms:
  - AdamW
related:
  - Optimizer
  - Stochastic Gradient Descent
  - Learning Rate
---
**Adam** is an adaptive optimizer that tracks exponential moving averages of gradients and squared gradients per parameter. **AdamW** decouples weight decay from the adaptive step and is preferred for transformers.

It converges reliably across many deep learning tasks but still needs a sensible LR and regularization.
