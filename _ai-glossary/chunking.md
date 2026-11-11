---
title: Chunking
related:
  - RAG
  - Embedding
  - Tokenizer
  - Dense Retrieval
---
**Chunking** splits long documents into smaller passages for [embedding](/ai-glossary/embedding/) and retrieval. Chunk size and overlap trade context completeness against retrieval precision: too large and the vector is muddled; too small and answers span multiple hits.

### Why it matters
- Bad chunking is a top reason [RAG](/ai-glossary/rag/) misses the right paragraph or “hallucinates”
- Boundaries interact with the [tokenizer](/ai-glossary/tokenizer/) and embedding model’s useful length
- Structure beats naive fixed windows for manuals, tickets, and code

### How it shows up
Prefer splits on headings, chapters, or semantic boundaries; add modest overlap so sentences aren’t cut mid-thought. Store chunk IDs and source metadata for [citations](/ai-glossary/citation/). Re-tune chunking when you change the embedding model—what was optimal for one encoder often isn’t for another.
