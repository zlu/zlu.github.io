---
title: Dense Retrieval
related:
  - Sparse Retrieval
  - Embedding
  - RAG
  - Approximate Nearest Neighbor
---
**Dense retrieval** encodes queries and documents into vectors with an [embedding](/ai-glossary/embedding/) model, then finds nearest neighbors—usually via [ANN](/ai-glossary/approximate-nearest-neighbor/) indexes. Dual-encoders keep document embeddings offline so online work is one query encode + search.

### Why it matters
- Captures paraphrase and semantic match that [BM25](/ai-glossary/bm25/)/sparse misses
- Core first stage in many [RAG](/ai-glossary/rag/) pipelines
- Quality hinges on embedding model, negatives, and [chunking](/ai-glossary/chunking/)

### Relation to sparse retrieval
Sparse methods win on exact tokens and rare IDs; dense wins on meaning. Hybrid + rerank is often best. Failures look like “wrong neighbors”: fix the encoder, chunks, or metric before blaming the LLM generator.
