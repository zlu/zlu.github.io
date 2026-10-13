---
title: Dropout
related:
  - Regularization
  - Overfitting
---
**Dropout** randomly zeroes activations during training so the network cannot rely on brittle co-adaptations. At evaluation time, dropout is disabled and weights are scaled appropriately.

Transformers often use modest dropout on attention and MLP blocks; fine-tuning sometimes lowers it.
