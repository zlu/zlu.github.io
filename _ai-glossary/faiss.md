---
title: FAISS
related:
  - Approximate Nearest Neighbor
  - Vector Database
  - Embedding
  - HNSW
---
**FAISS** (Facebook AI Similarity Search) is a library of exact and approximate vector search indexes on CPU/GPU. It is a common building block under custom [RAG](/ai-glossary/rag/) stacks and inside some [vector databases](/ai-glossary/vector-database/).

### Index families
- **Flat** — exact search; great recall, limited scale
- **IVF** — coarse quantizer + inverted lists for speed
- **HNSW** — graph-based [ANN](/ai-glossary/approximate-nearest-neighbor/) with strong recall
- **PQ / OPQ** — product quantization for memory compression

### Choosing a config
Trade memory, latency, and recall against corpus size. GPU indexes help at large $n$; CPU [HNSW](/ai-glossary/hnsw/) is often enough for mid-scale apps. Always measure recall@k on a labeled query set before locking the index type.

FAISS handles search math; you still own chunking, [embedding models](/ai-glossary/embedding-model/), and metadata filters around it.
