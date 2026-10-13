---
title: Layer Normalization
synonyms:
  - LayerNorm
related:
  - Transformer
  - Residual Connection
---
**Layer normalization** re-centers and re-scales activations across features for each example, stabilizing deep transformer training.

Pre-Norm vs Post-Norm placement changes training dynamics; most modern LLMs use Pre-Norm (or RMSNorm variants).
