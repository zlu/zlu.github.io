---
title: Decoder-Only
related:
  - Encoder–Decoder
  - Causal Language Model
  - Transformer
---
**Decoder-only** models (GPT-style) stack [Transformer](/ai-glossary/transformer/) blocks with a causal mask so each [token](/ai-glossary/token/) attends only leftward. The same stack handles [pretraining](/ai-glossary/pretraining/) and left-to-right generation—there is no separate bidirectional encoder.

### Why it matters
- Dominates open and closed LLM deployments
- Scaling and serving are simpler than [encoder–decoder](/ai-glossary/encoder-decoder/) for many generative workloads
- Unified next-token objective aligns training with open-ended generation

### Relation to other layouts
[Causal LMs](/ai-glossary/causal-language-model/) are typically decoder-only. Encoder–decoder (T5-style) still helps some seq2seq and span-corruption setups; masked encoders help understanding/retrieval. When someone says “LLM” in 2020s product talk, they almost always mean a decoder-only Transformer.
