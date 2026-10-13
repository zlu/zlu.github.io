---
title: Decoder-Only
related:
  - Encoder–Decoder
  - Causal Language Model
  - Transformer
---
**Decoder-only** models (GPT-style) apply a causal mask so tokens attend only leftward. The same stack handles pretraining and generation.

They dominate open and closed LLM deployments due to scaling simplicity and unified training.
