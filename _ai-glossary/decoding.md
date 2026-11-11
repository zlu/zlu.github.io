---
title: Decoding
related:
  - Sampling
  - Beam Search
  - Inference
  - Temperature
---
**Decoding** is the algorithm that turns model logits into a concrete [token](/ai-glossary/token/) sequence at [inference](/ai-glossary/inference/) time: greedy argmax, [sampling](/ai-glossary/sampling/) (with [temperature](/ai-glossary/temperature/), top-$k$, top-$p$), [beam search](/ai-glossary/beam-search/), contrastive search, and constrained / grammar-guided variants.

### Why it matters
- Same weights can feel “creative” or “deterministic” depending on the decoder
- Task fit beats defaults: chat ≠ translation ≠ JSON schema fill
- Latency and diversity trade off with beam width and sampling settings

### How it shows up
APIs expose temperature, top-p, stop sequences, and max tokens—those *are* decoding knobs. Evaluate with task metrics (exact match, rubric, pass@k), not only [perplexity](/ai-glossary/perplexity/). Constrained decoding helps structured outputs; unrestricted sampling helps open-ended prose.
