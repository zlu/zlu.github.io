---
title: RAG
synonyms:
  - Retrieval-Augmented Generation
related:
  - Vector Database
  - Embedding
  - Chunking
  - Reranking
  - Grounding
---
**Retrieval-augmented generation (RAG)** retrieves relevant documents at query time and conditions the LLM on that context, instead of relying only on parametric memory.

Typical pipeline: chunk corpus → embed → index → retrieve top-$k$ → (optional rerank) → prompt the model with passages. RAG improves factual grounding and updatability; CPT/SFT still help the model *use* retrieved text better.
