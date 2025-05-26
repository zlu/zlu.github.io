---
title: Word Embedding
---
A word embedding is a type of embedding specifically used in Natural Language Processing (NLP). It maps words (or subwords) to real-valued vectors in a continuous vector space, where semantically similar words are close together.

### Example word embeddings:
- Word2Vec
- GloVe
- FastText
- BERT (contextual embeddings)

### Properties:
- Vectors are typically 50 to 1,024 dimensions
- Similar meanings → similar vectors (cosine similarity)

### Example:
```python
word_vectors["king"] - word_vectors["man"] + word_vectors["woman"] ≈ word_vectors["queen"]
```
See: Cosine Similarity
