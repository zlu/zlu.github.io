---
title: Logit
related:
  - Softmax
  - Temperature
  - Cross-Entropy
---
A **logit** is a raw, unnormalized score output by a model before softmax (or similar) converts scores to probabilities.

Temperature scaling divides logits before softmax; cross-entropy training operates on logits via log-softmax for numerical stability.
