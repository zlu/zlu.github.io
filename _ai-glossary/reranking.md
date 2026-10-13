---
title: Reranking
synonyms:
  - Reranker
related:
  - RAG
  - Dense Retrieval
  - Semantic Search
---
**Reranking** rescores an initial candidate set of retrieved passages with a stronger (often cross-encoder) model so the LLM sees the best evidence first.

A cheap retriever proposes top-50; a reranker keeps top-5. This two-stage pattern is a high-leverage RAG upgrade when recall is decent but precision at $k$ is weak.
