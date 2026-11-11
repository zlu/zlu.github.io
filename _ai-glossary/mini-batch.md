---
title: Mini-Batch
related:
  - Batch Size
  - Stochastic Gradient Descent
  - Gradient Accumulation
  - Epoch
  - Learning Rate
---
A **mini-batch** is a small subset of training examples used to estimate the gradient in one optimizer step. Full-batch GD is rare for deep nets; pure single-example SGD is noisy and underuses accelerators.

### Why batch
- Better GPU utilization (larger matmuls)
- Lower variance gradients than one example
- Effective [batch size](/ai-glossary/batch-size/) interacts with [learning rate](/ai-glossary/learning-rate/)

### Practice
Tokens-per-step (examples × sequence length) matters more than “batch = 32” alone. When the GPU cannot hold a large batch, use [gradient accumulation](/ai-glossary/gradient-accumulation/) to simulate one. An [epoch](/ai-glossary/epoch/) is one full pass over the dataset via successive mini-batches.
