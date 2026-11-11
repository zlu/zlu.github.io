---
title: Word Embedding
synonyms:
  - Static Embedding
related:
  - Embedding
  - Cosine Similarity
  - Tokenizer
  - Representation Learning
  - Contrastive Learning
---
A **word embedding** maps a word (or subword) to a real-valued vector so that semantic similarity becomes geometric closeness in the vector space. Classic static methods assign one vector per vocabulary item; contextual models (BERT-style) produce a different vector for the same word depending on its sentence.

### Classic static methods
- **Word2Vec** — skip-gram / CBOW predictive training
- **GloVe** — global co-occurrence factorization
- **FastText** — subword n-grams help rare and morphologically rich words

### Properties
- Typical width: tens to hundreds of dimensions (modern LLM [embeddings](/ai-glossary/embedding/) are larger)
- Nearby vectors ↔ related meaning or usage ([cosine similarity](/ai-glossary/cosine-similarity/))
- Linear analogies were a famous (if brittle) demo: `king − man + woman ≈ queen`

### How it shows up today
LLM stacks still begin with an embedding matrix over [tokenizer](/ai-glossary/tokenizer/) ids, then contextualize those vectors with attention. Retrieval systems often embed passages with a dedicated [embedding model](/ai-glossary/embedding-model/) rather than raw Word2Vec. Prefer contextual or instruction-tuned embedders for search; keep static word vectors for pedagogy, lexicons, and lightweight features.
