---
title: Vector Database
synonyms:
  - Vector DB
  - Vector Store
related:
  - Embedding
  - RAG
  - Semantic Search
  - Approximate Nearest Neighbor
  - FAISS
  - HNSW
  - Dense Retrieval
---
A **vector database** stores high-dimensional [embeddings](/ai-glossary/embedding/) and supports similarity search—often via [approximate nearest neighbor](/ai-glossary/approximate-nearest-neighbor/) indexes such as [HNSW](/ai-glossary/hnsw/). It is the retrieval backend for many [RAG](/ai-glossary/rag/) and [semantic search](/ai-glossary/semantic-search/) systems.

### What it usually provides
- Insert / update / delete of vectors plus payload metadata
- Similarity queries (cosine, L2, inner product)
- Metadata filters (tenant, date, ACL tags)
- Optional hybrid sparse+dense search
- Collections/namespaces for multi-tenant apps

Libraries like [FAISS](/ai-glossary/faiss/) are indexes you embed in your process; managed vector DBs add durability, replication, and ops APIs. Index choice, dimensionality, and filter selectivity matter more than marketing labels—measure recall@k and latency on your corpus.
