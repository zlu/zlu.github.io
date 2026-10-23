---
title: Attention
synonyms:
  - Soft Attention
  - Attention Mechanism
related:
  - Self-Attention
  - Multi-Head Attention
  - Encoder–Decoder
  - Encoder
  - Decoder
  - Transformer
  - Softmax
---
**Attention** lets a model decide, at each step, which parts of a memory of vectors matter most—and then form a weighted average of those vectors.

In the original soft-attention encoder–decoder setup (Bahdanau et al., 2015), the decoder compares its current state to every encoder state, turns the scores into weights with a softmax, and blends the encoder states into a *context vector*. That removed the need to crush the whole source into one fixed vector.

Self-attention in Transformers uses the same skeleton inside one sequence: queries, keys, and values are projections of the tokens themselves.

$$\text{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$
