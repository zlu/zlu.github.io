---
title: Normalization
related:
  - Layer Normalization
  - Feature Engineering
  - Residual Connection
  - Mixed Precision
---
**Normalization** rescales data or activations so magnitudes stay comparable — e.g., z-score / min-max on features, or LayerNorm / RMSNorm inside networks.

### Where it appears
- **Input / feature norm** — part of [feature engineering](/ai-glossary/feature-engineering/); helps classical models and shallow nets
- **BatchNorm** — stats over the batch (common in vision; rare in LLMs)
- **[Layer normalization](/ai-glossary/layer-normalization/) / RMSNorm** — per-token stats; standard in transformers, paired with [residual connections](/ai-glossary/residual-connection/)

### Why it matters
Better-conditioned optimization, more stable deep stacks, and friendlier behavior under [mixed precision](/ai-glossary/mixed-precision/). The right form depends on whether you normalize inputs, batches, or layers — don’t mix them up when debugging training.
