---
title: Context Window
synonyms:
  - Context Length
related:
  - Token
  - KV Cache
  - Inference
---
The **context window** is the maximum number of tokens a model can attend to in one forward pass (prompt + generation, depending on API semantics).

Longer windows help with documents and agents but raise memory (KV cache) and latency. Effective use still depends on attention quality — stuffing irrelevant text can hurt more than help.
