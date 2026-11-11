---
title: Sampling
related:
  - Temperature
  - Top-p
  - Top-k
  - Decoding
  - Softmax
  - Beam Search
---
**Sampling** draws the next token from the model’s probability distribution (after optional shaping), introducing randomness into generation. Contrast with greedy decode or [beam search](/ai-glossary/beam-search/), which are deterministic or search-based [decoding](/ai-glossary/decoding/).

### Common knobs
- **[Temperature](/ai-glossary/temperature/)** — sharpen or flatten the [softmax](/ai-glossary/softmax/) distribution
- **[Top-$k$](/ai-glossary/top-k/)** — keep only the $k$ highest-probability tokens
- **[Top-$p$](/ai-glossary/top-p/)** (nucleus) — keep the smallest set whose mass ≥ $p$

### Practice
Low randomness → more deterministic, repetitive text. High randomness → more diverse, sometimes incoherent text. Prefer lower temperature / tighter nucleus for code and factual QA; allow more entropy for brainstorming. Fix seeds when you need reproducible evals.
