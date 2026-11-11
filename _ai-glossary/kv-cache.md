---
title: KV Cache
synonyms:
  - Key-Value Cache
related:
  - Inference
  - Context Window
  - Attention
  - Speculative Decoding
---
During autoregressive decoding, attention keys and values for past tokens are reused. The **KV cache** stores them so the model does not recompute the full prompt every step—critical for [inference](/ai-glossary/inference/) latency.

### Memory footprint
Cache size grows with batch × layers × heads × sequence × dtype. For long [context windows](/ai-glossary/context-window/), KV memory often binds serving capacity before weight VRAM does.

### Mitigations
- **GQA / MQA** — fewer KV heads than query heads
- **Quantized KV** — lower-precision cache tensors
- **Paging** — e.g., PagedAttention for fragmented batching
- **Eviction / sliding windows** — drop or compress old tokens when allowed

Prefill writes the cache for the prompt; decode appends one step at a time. Tuning max context and batch size is as much a KV problem as a compute problem.
