---
title: Speculative Decoding
related:
  - Inference
  - Decoding
---
**Speculative decoding** uses a small draft model (or draft heads) to propose several tokens that a larger target model verifies in parallel, reducing wall-clock latency without changing the target distribution (when done correctly).

It is a serving optimization, not a training method.
