---
title: Layer Normalization
synonyms:
  - LayerNorm
related:
  - Transformer
  - Residual Connection
  - Normalization
---
**Layer normalization** re-centers and re-scales activations across features for each example, stabilizing deep [transformer](/ai-glossary/transformer/) training. Unlike batch norm, it does not depend on batch statistics—friendly to variable-length NLP batches.

### Pre-Norm vs Post-Norm
- **Post-Norm** — Norm after the residual add (original Transformer)
- **Pre-Norm** — Norm before sublayers; usually easier to train at depth
Most modern LLMs use Pre-Norm or close variants such as **RMSNorm** (scale without mean-centering).

### Why it matters
Norm placement changes gradient scale through [residual connections](/ai-glossary/residual-connection/) and affects how aggressively you can set [learning rate](/ai-glossary/learning-rate/). When porting architectures, match the norm type and position—not only hidden size.

Learned scale/bias (or RMSNorm scale alone) are small but important parameters; do not drop them when converting checkpoints.
