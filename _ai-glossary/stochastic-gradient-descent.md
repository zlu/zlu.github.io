---
title: Stochastic Gradient Descent
synonyms:
  - SGD
related:
  - Gradient Descent
  - Optimizer
  - Mini-Batch
---
**Stochastic gradient descent** estimates the gradient on a subset of data (a mini-batch) rather than the full dataset each step. Noise from mini-batches can help escape sharp minima and scales training to large corpora.

With momentum and schedules, SGD remains competitive; Adam-family methods are more common for LLMs.
