---
title: Embedding Model
related:
  - Embedding
  - Semantic Search
  - RAG
---
An **embedding model** maps text (or other modalities) to fixed-size vectors optimized for similarity tasks. It is usually separate from the generative LLM in a RAG stack.

Domain drift matters: embed with a model trained for retrieval, and re-index if you change models.
