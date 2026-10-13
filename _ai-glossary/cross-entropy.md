---
title: Cross-Entropy
related:
  - Loss Function
  - Softmax
  - Perplexity
---
**Cross-entropy loss** for classification compares the predicted probability distribution with the one-hot (or soft) target. For language modeling, it is the negative log probability of the correct next token.

Perplexity is the exponential of average cross-entropy. Softmax + cross-entropy is the default decoding head training setup for classifiers and LMs.
