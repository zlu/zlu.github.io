---
title: ReLU
synonyms:
  - Rectified Linear Unit
related:
  - Activation Function
---
**ReLU** is $f(x) = \max(0, x)$. It is simple, sparse, and historically important for deep nets. Variants (Leaky ReLU, GELU) address dead units or smooth the kink at zero.

Many modern LLMs prefer GELU/SwiGLU over plain ReLU in feed-forward layers.
