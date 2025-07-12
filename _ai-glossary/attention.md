---
title: Attention
related:
  - Transformer
---
The attention mechanism is a technique used in machine learning - especially in the natural language process (NLP) and computer vision - that allows models to focus on the most relevant parts of the input data when making decisions or predictions.

The core idea is to treat all parts of the input equally, attention assigns weights to different elements, indicating their importance for a given task.  These weights are learned during training.


In the context of the Transformer architecture (e.g., GPT, BERT):
Let:
- Q: Query
- K: Key
- V: Value

The Scaled Dot-Product Attention is:

$\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V$
- $QK^T$: Measures relevance between query and key.
- $\sqrt{d_k}$: Scaling factor to stabilize gradients.
- $softmax$: Converts scores to probabilities (attention weights).
- The result is a weighted sum of the values V, emphasizing relevant parts.
