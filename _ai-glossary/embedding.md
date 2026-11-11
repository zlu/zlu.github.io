---
title: Embedding
related:
  - Word Embedding
  - Matrix Embedding
  - Token
  - LM Head
  - Vector Database
  - Dense Retrieval
  - Cosine Similarity
---
An **embedding** is a learned representation of data as a dense continuous vector. It maps high-dimensional, discrete, or symbolic inputs (words, tokens, users, items, images) into a space where geometry—distance and direction—reflects semantic or structural similarity.

In a language model, the embedding table maps each [token](/ai-glossary/token/) id to a vector that enters the Transformer stack. The final [LM head](/ai-glossary/lm-head/) maps hidden vectors back to vocabulary logits (sometimes with tied weights). For retrieval, a dedicated embedding model encodes queries and passages for [dense retrieval](/ai-glossary/dense-retrieval/) in a [vector database](/ai-glossary/vector-database/).

### Why use embeddings
- Compress discrete symbols into fixed-size vectors
- Enable similarity search ([cosine](/ai-glossary/cosine-similarity/), dot product)
- Transfer structure learned on large corpora to downstream tasks

See: [word embedding](/ai-glossary/word-embedding/), [matrix embedding](/ai-glossary/matrix-embedding/). Training objectives (masked LM, contrastive pairs, next-token) shape what “nearby” means in the space.
