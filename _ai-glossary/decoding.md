---
title: Decoding
related:
  - Sampling
  - Beam Search
  - Inference
  - Temperature
---
**Decoding** is the algorithm that turns model logits into a token sequence: greedy argmax, sampling, beam search, contrastive search, and constrained decoding variants.

There is no universal best decoder — match the method to the task (translation vs chat vs structured JSON) and evaluate with task metrics, not only perplexity.
