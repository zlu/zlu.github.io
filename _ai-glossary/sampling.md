---
title: Sampling
related:
  - Temperature
  - Top-p
  - Top-k
  - Decoding
---
**Sampling** draws the next token from the model’s probability distribution (after optional temperature / top-$k$ / top-$p$ shaping), introducing randomness into generation.

Low randomness → more deterministic, repetitive text. High randomness → more diverse, sometimes incoherent text. Choose sampling settings per task (creative writing vs code vs factual QA).
