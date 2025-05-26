---
title: Matrix Embedding
---
A matrix embedding refers to stacking multiple embeddings into a matrix form. This is common when dealing with sequences like:
- Sentences (word embeddings stacked into a 2D matrix)
- Paragraphs (sentence embeddings stacked)
- Users/items in recommender systems

### Shape example:
If you have a sentence of 10 words and each word embedding is 300-dimensional, the sentence embedding matrix is:
```python
shape = (10, 300)
```
