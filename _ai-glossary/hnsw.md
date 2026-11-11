---
title: HNSW
synonyms:
  - Hierarchical Navigable Small World
related:
  - Approximate Nearest Neighbor
  - Vector Database
  - FAISS
  - Dense Retrieval
---
**HNSW** is a popular graph-based [ANN](/ai-glossary/approximate-nearest-neighbor/) index: multi-layer proximity graphs enable roughly logarithmic-time search with strong recall. It is widely used inside [vector databases](/ai-glossary/vector-database/) and libraries like [FAISS](/ai-glossary/faiss/) for dense retrieval at moderate-to-large scale.

### Intuition
Upper layers are sparse “highways” for long jumps; lower layers refine neighbors locally. Search enters at the top, greedily walks toward the query, then descends layers for finer candidates.

### Knobs
- **M** — connections per node (memory vs recall)
- **efConstruction / efSearch** — build and query candidate list sizes (speed vs quality)

HNSW is in-memory heavy compared with compressed IVF-PQ; pick it when recall and latency matter more than minimal RAM. Always validate recall@k on your embedding space—graph quality is data-dependent.
