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
  - Flash Attention
---
**Attention** lets a model decide, at each step, which parts of a memory of vectors matter most—and then form a weighted average of those vectors.

In the original soft-attention encoder–decoder setup (Bahdanau et al., 2015), the [decoder](/ai-glossary/decoder/) compares its current state to every [encoder](/ai-glossary/encoder/) state, turns the scores into weights with a [softmax](/ai-glossary/softmax/), and blends the encoder states into a *context vector*. That removed the need to crush the whole source into one fixed vector.

[Self-attention](/ai-glossary/self-attention/) in [Transformers](/ai-glossary/transformer/) uses the same skeleton inside one sequence: queries, keys, and values are projections of the tokens themselves. Cross-attention keeps queries on one side (decoder) and keys/values on the other (encoder).

$$\text{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

The $\sqrt{d_k}$ scale keeps dot products from growing with dimension so the softmax does not saturate. [Multi-head attention](/ai-glossary/multi-head-attention/) runs several of these in parallel; efficient kernels like [FlashAttention](/ai-glossary/flash-attention/) make long sequences practical.
