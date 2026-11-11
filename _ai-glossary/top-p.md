---
title: Top-p
synonyms:
  - Nucleus Sampling
related:
  - Top-k
  - Temperature
  - Sampling
  - Decoding
  - Softmax
---
**Top-$p$ (nucleus) sampling** keeps the smallest set of tokens whose cumulative probability mass exceeds $p$ (e.g., 0.9), then samples from that set—usually after [temperature](/ai-glossary/temperature/) scaling.

It adapts the candidate set to the distribution’s shape: tighter when the model is confident (mass on few tokens), wider when uncertain. That is why it is often preferred over fixed [top-$k$](/ai-glossary/top-k/) alone.

### Knobs
- $p \approx 0.9$–$0.95$ is a common chat default
- Lower $p$ → safer, more conservative text
- Combine with temperature; do not crank both wildly at once

Nucleus sampling is a [decoding](/ai-glossary/decoding/) / [sampling](/ai-glossary/sampling/) choice at inference time—it does not change trained weights. Measure task metrics (factuality, diversity, format adherence) when you retune $p$.
