---
title: Logit
related:
  - Softmax
  - Temperature
  - Cross-Entropy
  - LM Head
---
A **logit** is a raw, unnormalized score output by a model before [softmax](/ai-glossary/softmax/) (or similar) converts scores to probabilities. In LMs, the [LM head](/ai-glossary/lm-head/) emits one logit per vocabulary token at each position.

### Decoding and training
[Temperature](/ai-glossary/temperature/) scaling divides logits before softmax: higher $T$ flattens the distribution; $T \to 0$ approaches argmax. [Cross-entropy](/ai-glossary/cross-entropy/) training operates on logits via log-softmax for numerical stability—you rarely softmax explicitly in the loss path.

### Intuition
Larger logit ⇒ higher probability after normalization, but absolute logit scale is meaningless without the rest of the vector. Compare logits within one distribution; do not treat a single logit as a calibrated confidence unless you calibrate.

Top-k / top-p sampling truncate or renormalize over logits (or their softmax) before drawing the next token.
