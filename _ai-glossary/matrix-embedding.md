---
title: Matrix Embedding
description: "A matrix embedding stacks token or item vectors into a 2D matrix (sequence length × dimension), the usual batch input shape for transformers."
keywords: "matrix embedding, embedding matrix, word embedding matrix, sequence embedding"
related:
  - Embedding
  - Word Embedding
  - Tensor
  - Token
  - Transformer
---
A **matrix embedding** (often called an embedding matrix in code) stacks multiple vectors into a 2D array. Typical cases:

- A sentence: word embeddings stacked to `(seq_len, dim)`
- A paragraph: sentence vectors stacked
- Recommenders: user/item embedding tables as large matrices
- A mini-batch: often `(batch, seq_len, dim)`—a 3D [tensor](/ai-glossary/tensor/) of stacked matrices

### Shape example

If a sentence has 10 words and each word embedding is 300-dimensional:

```python
shape = (10, 300)
```

In Transformers, the token [embedding](/ai-glossary/embedding/) lookup produces exactly this kind of matrix (plus position), which then flows through attention and MLP blocks. Related: [word embedding](/ai-glossary/word-embedding/), [tensor](/ai-glossary/tensor/).
