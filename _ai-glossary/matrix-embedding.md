---
title: Matrix Embedding (Embedding Matrix)
description: "Matrix embedding vs embedding matrix: stacked token vectors (seq_len × dim) after lookup, or the |V| × dim weight table itself — shapes and when each term appears."
keywords: "matrix embedding, embedding matrix, embedded matrix, word embedding matrix, sequence embedding, embedding table"
related:
  - Embedding
  - Word Embedding
  - Tensor
  - Token
  - Transformer
  - Soft Prompt
---
A **matrix embedding** (often called an **embedding matrix** in code; searchers also type **embedded matrix**) stacks multiple vectors into a 2D array. Typical cases:

- A sentence: word embeddings stacked to `(seq_len, dim)`
- A paragraph: sentence vectors stacked
- Recommenders: user/item embedding tables as large matrices
- A mini-batch: often `(batch, seq_len, dim)`, a 3D [tensor](/ai-glossary/tensor/) of stacked matrices

People also say "embedding matrix" for the **lookup table** itself: a learned `(|V|, dim)` weight matrix where each row is one [token](/ai-glossary/token/)'s vector. Both meanings show up in papers and code; the shape tells you which one.

### Shape example

If a sentence has 10 words and each word embedding is 300-dimensional:

```python
# stacked token vectors for one sequence
shape = (10, 300)

# vocabulary lookup table (different object, same name in some codebases)
vocab_table = (50_000, 300)
```

In Transformers, the token [embedding](/ai-glossary/embedding/) lookup reads rows from the vocab table and stacks them into the sequence matrix (then adds position). That matrix flows through attention and MLP blocks. Related: [word embedding](/ai-glossary/word-embedding/), [tensor](/ai-glossary/tensor/).

### Matrix embedding vs soft prompt

A normal embedding matrix maps discrete token IDs to vectors. A [soft prompt](/ai-glossary/soft-prompt/) is also a small matrix of vectors, but it is **not** tied to vocabulary rows: it is prepended as extra learned positions during [prompt tuning](/ai-glossary/prompt-tuning/), while the base embedding table usually stays frozen.

### Why the name shows up in search

Writers mix "matrix embedding" and "embedding matrix" for the same shapes. If you are debugging `(batch, seq, dim)` mismatches or LoRA/embedding-table edits, you almost always want the stacked sequence matrix or the `|V| × dim` lookup weights, not a separate third concept.
