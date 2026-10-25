---
title: Autoregressive
synonyms:
  - Autoregressive Model
  - AR
related:
  - Causal Language Model
  - Next-Token Prediction
  - Decoder-Only
  - Teacher Forcing
  - Inference
---
An **autoregressive** model builds a sequence **one step at a time**, and each new token is predicted only from what came **before** it—not from the future.

For language models that means: to predict token $t+1$, use tokens $1,\ldots,t$ only. Decoder-only LLMs (GPT-style) are autoregressive; a causal mask on [self-attention](/ai-glossary/self-attention/) enforces that rule during training so the model cannot cheat by peeking ahead. The French decoder in encoder–decoder MT is also autoregressive when generating.

Inference cost scales with output length because each new token needs a forward pass (amortized with KV caching).
