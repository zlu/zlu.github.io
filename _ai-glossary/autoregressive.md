---
title: Autoregressive
related:
  - Causal Language Model
  - Next-Token Prediction
  - Inference
---
An **autoregressive** model generates a sequence one token at a time, conditioning each step on previously generated tokens. Decoder-only LLMs are autoregressive language models.

Inference cost scales with output length because each new token requires a forward pass (amortized with KV caching).
