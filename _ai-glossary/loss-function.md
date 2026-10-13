---
title: Loss Function
synonyms:
  - Objective Function
related:
  - Cross-Entropy
  - Gradient Descent
  - Evaluation
---
A **loss function** quantifies prediction error during training. Optimization minimizes expected loss on the training distribution.

For LLMs, token-level **cross-entropy** (negative log-likelihood) is standard. Downstream quality still needs separate task metrics — loss is a proxy, not the product goal.
