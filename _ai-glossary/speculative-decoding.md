---
title: Speculative Decoding
related:
  - Inference
  - Decoding
  - Sampling
  - KV Cache
---
**Speculative decoding** speeds up LLM [inference](/ai-glossary/inference/) without changing the target model’s distribution (when implemented correctly). A small *draft* model (or draft heads) proposes several tokens ahead; the large *target* model verifies them in one parallel forward pass and accepts a prefix of matching tokens.

Accepted drafts amortize the expensive target forward; rejections fall back to the target’s own next-token choice. The draft must be cheap and reasonably aligned with the target—otherwise accept rates tank and you gain nothing.

### Why it helps
- Autoregressive generation is memory-bandwidth bound; verifying $k$ tokens together uses the same weights more efficiently
- Works with greedy or sampling [decoding](/ai-glossary/decoding/) under the right acceptance rules
- Orthogonal to quantization and [KV cache](/ai-glossary/kv-cache/) tricks

It is a serving optimization, not a training method. Expect larger wins when the draft is much faster than the target and the task is locally predictable (code, boilerplate, structured text).
