---
title: Optimizer
related:
  - Adam
  - Gradient Descent
  - Stochastic Gradient Descent
  - Learning Rate
---
An **optimizer** applies gradients to update parameters: SGD, Momentum, Adam, AdamW, and many variants.

AdamW is the default for most transformer training. Choice interacts with LR schedule, weight decay, and batch size — tune as a system, not in isolation.
