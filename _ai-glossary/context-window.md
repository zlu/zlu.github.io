---
title: Context Window
synonyms:
  - Context Length
related:
  - Token
  - KV Cache
  - Inference
---
The **context window** is the maximum number of [tokens](/ai-glossary/token/) a model can attend to in one forward pass—prompt plus generation, depending on API accounting. It caps how much document, chat history, and tool output you can stuff into a single call.

### Why it matters
- Longer windows enable long docs and multi-step agents
- Memory ([KV cache](/ai-glossary/kv-cache/)) and latency grow with filled context
- “Supports 128k” ≠ “uses 128k well”—attention quality and position schemes vary

### How it shows up
At [inference](/ai-glossary/inference/), overflowing the window truncates or errors. Effective systems still retrieve and summarize rather than dump everything: irrelevant tokens can hurt more than help. Measure useful recall inside the window, not just the marketed maximum.
