---
title: Approximate Nearest Neighbor
synonyms:
  - ANN
related:
  - Vector Database
  - HNSW
  - FAISS
  - Dense Retrieval
---
**Approximate nearest neighbor (ANN)** search finds mostly-the-nearest vectors much faster than exact scan. Indexes such as [HNSW](/ai-glossary/hnsw/), IVF, and LSH (via [FAISS](/ai-glossary/faiss/) or a [vector database](/ai-glossary/vector-database/)) trade a tunable amount of recall for large latency wins at corpus scale.

### Why it matters
- Required for practical [dense retrieval](/ai-glossary/dense-retrieval/) over millions+ embeddings
- Recall@k vs latency is a product knob, not a fixed property of “ANN”
- Bad index params or metric mismatch (cosine vs L2) quietly tank RAG quality

### How it shows up
Build embeddings → choose metric → tune `efSearch` / nprobe / etc. on a held-out query set. Evaluate with *your* embedding distribution; synthetic unit spheres mislead. Exact brute force remains useful for small sets and for measuring how much recall ANN costs.
