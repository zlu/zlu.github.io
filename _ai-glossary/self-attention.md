---
title: Self-Attention
related:
  - Attention
  - Multi-Head Attention
  - Transformer
  - Positional Encoding
  - Flash Attention
  - Softmax
---
**Self-attention** lets each position in a sequence compute a weighted combination of all positions (including itself). Queries, keys, and values are linear projections of the *same* input sequence—unlike encoder–decoder cross-attention, where queries come from the decoder and keys/values from the encoder.

$$\text{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Unlike RNN recurrence, self-attention is fully parallelizable across the sequence length and can capture long-range dependencies in a single layer. Cost grows roughly as $O(n^2)$ in sequence length $n$, which motivates [FlashAttention](/ai-glossary/flash-attention/), sparse attention, and sliding windows.

### Variants you will meet
- **Causal / masked** — tokens cannot attend to the future ([autoregressive](/ai-glossary/autoregressive/) LMs)
- **Bidirectional** — full context (encoder / BERT-style)
- **Multi-head** — several subspaces in parallel ([multi-head attention](/ai-glossary/multi-head-attention/))

Because self-attention is permutation-invariant without positions, models add [positional encoding](/ai-glossary/positional-encoding/) (absolute, relative, or RoPE).
