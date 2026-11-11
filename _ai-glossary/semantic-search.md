---
title: Semantic Search
related:
  - Embedding
  - Vector Database
  - RAG
  - Dense Retrieval
  - Hybrid Search
  - Sparse Retrieval
---
**Semantic search** retrieves documents by meaning: a query and each passage are mapped to dense [embeddings](/ai-glossary/embedding/), then nearest neighbors are ranked by similarity (often cosine). Exact keyword overlap is optional, not required.

It shines on paraphrases (“heart attack” ↔ “myocardial infarction”) where classic keyword search fails. It struggles on rare IDs, exact quotes, and legal codes unless those strings appear in training or you add a sparse channel.

### Typical stack
1. Chunk and embed the corpus
2. Index vectors in a [vector database](/ai-glossary/vector-database/)
3. Embed the query and run ANN search
4. Optionally [rerank](/ai-glossary/reranking/) or fuse with [BM25](/ai-glossary/bm25/) ([hybrid search](/ai-glossary/hybrid-search/))

Semantic search is the retrieval half of most [RAG](/ai-glossary/rag/) pipelines; quality tracks the embedding model, chunking, and filters more than the DB brand.
