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
  - KV Cache
---
An **autoregressive** model builds a sequence **one step at a time**, and each new token is predicted only from what came **before** it—not from the future.

For language models that means: to predict token $t+1$, use tokens $1,\ldots,t$ only. Decoder-only LLMs (GPT-style) are autoregressive; a causal mask on [self-attention](/ai-glossary/self-attention/) enforces that rule during training so the model cannot cheat by peeking ahead. The French decoder in encoder–decoder MT is also autoregressive when generating.

### Train and serve
- Training uses [teacher forcing](/ai-glossary/teacher-forcing/) / [next-token prediction](/ai-glossary/next-token-prediction/) on gold prefixes
- [Inference](/ai-glossary/inference/) feeds the model its own outputs; cost scales with output length
- [KV cache](/ai-glossary/kv-cache/) amortizes attention over past tokens so each new step is cheaper

Autoregressive factorization $p(x)=\prod_t p(x_t\mid x_{<t})$ is what makes left-to-right LLMs both powerful and latency-sensitive at long generations.
