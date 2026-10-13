---
title: Multi-Head Attention
synonyms:
  - MHA
related:
  - Self-Attention
  - Attention
  - Transformer
---
**Multi-head attention** runs several attention operations in parallel (“heads”), each with its own projections. Outputs are concatenated and projected back to the model dimension.

Heads can specialize: some focus on local syntax, others on longer-range semantic links. Modern LLMs often use **grouped-query** or **multi-query** attention variants to reduce KV-cache cost at inference time while keeping most of the quality of full multi-head attention.
