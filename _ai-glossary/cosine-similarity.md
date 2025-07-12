---
title: Cosine Similarity
---
Cosine similarity is a metric used to measure how similar two vectors are by computing the cosine of the angle between them. It is widely used in machine learning, especially in text similarity, recommendation systems, and clustering.

### Intuition
- If two vectors point in exactly the same direction, their cosine similarity is 1.
- If they are orthogonal (completely different), the similarity is 0.
- If they point in opposite directions, the similarity is -1.

It ignores magnitude, focusing on orientation, which makes it great for comparing text embeddings where length may vary but direction (semantic meaning) matters.


### Formula

For two vectors A and B:

$ \text{cosine\_similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} $

Where:
- $A \cdot B$ = dot product of vectors A and B
- $\|A\|$ = Euclidean norm (length) of A
- $\|B\|$ = Euclidean norm of B


### Example (Python)
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

A = np.array([[1, 2, 3]])
B = np.array([[4, 5, 6]])

similarity = cosine_similarity(A, B)
print(similarity)  # Output: [[0.9746]]
```

### Use Cases
- NLP: comparing sentence or word embeddings
- Recommendation: finding similar users/items
- Clustering: grouping similar vectors
- Document similarity: e.g., search engines

