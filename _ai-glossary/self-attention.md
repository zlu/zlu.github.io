---
title: Self-Attention
related:
  - Attention
  - Multi-Head Attention
  - Transformer
---
**Self-attention** lets each position in a sequence compute a weighted combination of all positions (including itself). Queries, keys, and values are linear projections of the same input sequence.

$$\text{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Unlike RNN recurrence, self-attention is fully parallelizable across the sequence length and can capture long-range dependencies in a single layer. Cost grows roughly as $O(n^2)$ in sequence length $n$, which motivates techniques like FlashAttention, sparse attention, and sliding windows.
