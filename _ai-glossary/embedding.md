---
title: Embedding
related:
  - Word Embedding
  - Matrix Embedding
  - Token
  - LM Head
---
An **embedding** is a learned representation of data in a lower-dimensional space. It transforms high-dimensional, discrete, or symbolic data (like words, users, or items) into dense, continuous vectors that preserve semantic or structural relationships.

In a language model, the embedding table maps each [token](/ai-glossary/token/) id to a vector that enters the Transformer stack. The final [LM head](/ai-glossary/lm-head/) maps hidden vectors back to vocabulary logits (sometimes with tied weights).

Why use embeddings?
- Reduce dimensionality
- Enable similarity comparison
- Improve learning by preserving structure

See: [Word embedding](/ai-glossary/word-embedding/), [matrix embedding](/ai-glossary/matrix-embedding/).
