---
title: Embedding Model
related:
  - Embedding
  - Semantic Search
  - RAG
  - Dense Retrieval
---
An **embedding model** maps text (or other modalities) to fixed-size vectors optimized for similarity tasks. In a [RAG](/ai-glossary/rag/) stack it is usually separate from the generative LLM: one model indexes and retrieves; another reads the passages and answers.

### What “good” means
Vectors should place related queries and documents nearby under cosine or dot-product similarity. Quality depends on training objective (often [contrastive](/ai-glossary/contrastive-learning/)), hard negatives, chunk length alignment, and language/domain coverage.

### Operational rules
- Embed with a model trained for retrieval, not raw LM hidden states alone
- Re-index the whole corpus if you change models or pooling
- Domain drift hurts: fine-tune or swap embeddings when jargon and entities shift

Dimension, max tokens, and instruction-prefixed vs plain text APIs all affect recall—document them as part of the retrieval contract.
