---
title: Top-p
synonyms:
  - Nucleus Sampling
related:
  - Top-k
  - Temperature
  - Sampling
---
**Top-$p$ (nucleus) sampling** keeps the smallest set of tokens whose cumulative probability mass exceeds $p$ (e.g., 0.9), then samples from that set.

It adapts the candidate set to the distribution’s shape — tighter when the model is confident, wider when uncertain — and is often preferred over fixed top-$k$ alone.
