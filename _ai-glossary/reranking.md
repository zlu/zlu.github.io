---
title: Reranking
synonyms:
  - Reranker
related:
  - RAG
  - Dense Retrieval
  - Semantic Search
  - Sparse Retrieval
  - Hybrid Search
---
**Reranking** rescores an initial candidate set of retrieved passages with a stronger model (often a cross-encoder) so the LLM sees the best evidence first.

### Two-stage pattern
1. Cheap retriever ([dense](/ai-glossary/dense-retrieval/), sparse, or [hybrid](/ai-glossary/hybrid-search/)) proposes top-50 / top-100
2. Reranker keeps top-5–10 for the [RAG](/ai-glossary/rag/) prompt

### When it helps
High recall but weak precision at small $k$ — common when bi-encoders miss fine-grained relevance. Cost is extra latency and compute per query; cache or skip rerank for easy queries if needed. Always measure end-to-end answer quality, not only nDCG of the retriever.
