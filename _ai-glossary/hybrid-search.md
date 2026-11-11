---
title: Hybrid Search
related:
  - BM25
  - Dense Retrieval
  - Semantic Search
  - RAG
---
**Hybrid search** combines sparse and dense retrieval—typically [BM25](/ai-glossary/bm25/) plus [dense](/ai-glossary/dense-retrieval/) vectors—via score fusion or reciprocal rank fusion (RRF). You get lexical precision (IDs, rare tokens, exact phrases) and semantic recall (paraphrases, synonyms).

### Why it is a default
Enterprise text is messy: acronyms, part numbers, and proper nouns often break pure embedding search, while bag-of-words misses meaning. Hybrid recovers both failure modes for many [RAG](/ai-glossary/rag/) corpora.

### Implementation tips
- Fuse ranks (RRF) when score scales are incomparable
- Tune sparse/dense weights on a labeled query set
- Add metadata filters before or after fusion
- Optional [reranking](/ai-glossary/reranking/) on the fused top-$n$ for final precision

Measure end-to-end answer quality, not only first-stage recall—retrieval gains only help if the generator uses the passages.
