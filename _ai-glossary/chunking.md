---
title: Chunking
related:
  - RAG
  - Embedding
  - Tokenizer
---
**Chunking** splits long documents into smaller passages for embedding and retrieval. Chunk size and overlap trade off context completeness against retrieval precision.

Bad chunking (cutting mid-sentence, mixing unrelated sections, or making chunks too large) is a common reason RAG “hallucinates” or misses the right paragraph. Structure-aware splits (by heading, chapter, or semantic boundaries) usually beat naive fixed windows.
