---
title: Multi-Head Attention
synonyms:
  - MHA
related:
  - Self-Attention
  - Attention
  - Transformer
  - KV Cache
---
**Multi-head attention (MHA)** runs several [attention](/ai-glossary/attention/) operations in parallel (“heads”), each with its own $Q,K,V$ projections. Head outputs are concatenated and projected back to the model dimension.

### Why multiple heads
Different heads can specialize (local syntax vs longer semantic links). Splitting $d_{\text{model}}$ across $h$ heads keeps total compute similar to one wide head while increasing representational diversity.

### Inference variants
Full MHA stores keys/values per head in the [KV cache](/ai-glossary/kv-cache/). **Multi-query** and **grouped-query** attention share K/V across groups of query heads to cut cache size and bandwidth with little quality loss — standard in many recent LLMs. See also [self-attention](/ai-glossary/self-attention/) for the single-sequence case.
