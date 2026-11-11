---
title: Sparse Retrieval
related:
  - Dense Retrieval
  - BM25
  - Hybrid Search
  - Semantic Search
  - RAG
---
**Sparse retrieval** scores documents with high-dimensional sparse signals—typically term frequencies and document statistics ([BM25](/ai-glossary/bm25/), TF–IDF). Each document is a bag of weighted terms; most dimensions are zero.

It is strong on exact keywords, product IDs, error codes, and rare proper nouns that dense embeddings may smear. It is weak on paraphrase and synonym matching unless those surface forms appear in the query.

### Role in modern stacks
- Baseline lexical search for corpora with precise terminology
- Complementary channel in [hybrid search](/ai-glossary/hybrid-search/) fused with [dense retrieval](/ai-glossary/dense-retrieval/)
- Fast inverted-index lookup without a GPU embedding pass

In [RAG](/ai-glossary/rag/), sparse + dense fusion often beats either alone: sparse catches the literal match; dense catches the meaning.
