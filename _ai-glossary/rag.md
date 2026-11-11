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
  - Dense Retrieval
  - Hybrid Search
---
**Retrieval-augmented generation (RAG)** retrieves relevant documents at query time and conditions the LLM on that context, instead of relying only on parametric memory.

### Typical pipeline
1. [Chunk](/ai-glossary/chunking/) the corpus and embed passages
2. Index in a [vector database](/ai-glossary/vector-database/) (often plus keyword / [hybrid search](/ai-glossary/hybrid-search/))
3. Retrieve top-$k$ ([dense](/ai-glossary/dense-retrieval/) and/or sparse)
4. Optional [reranking](/ai-glossary/reranking/)
5. Prompt the model with selected passages for [grounding](/ai-glossary/grounding/)

### Tradeoffs
Improves factual updatability and citation potential; fails when retrieval misses, chunks are wrong, or the model ignores context. CPT/SFT still help the model *use* retrieved text. Evaluate retrieval and generation separately.
