---
title: KV Cache
synonyms:
  - Key-Value Cache
related:
  - Inference
  - Context Window
  - Attention
---
During autoregressive decoding, attention keys and values for past tokens are reused. The **KV cache** stores them so the model does not recompute the full prompt every step.

Cache size grows with batch × layers × heads × sequence × dtype — often the binding constraint for long-context serving. Quantized KV, paging (e.g., PagedAttention), and GQA/MQA reduce this footprint.
