---
title: LM Head
synonyms:
  - Final Vocab Projection
  - Vocabulary Projection
  - Output Projection
  - Unembedding
related:
  - Embedding
  - Logit
  - Softmax
  - Causal Language Model
  - Next-Token Prediction
---
The **LM head** (final vocab projection) is the last linear layer of a causal language model. It maps each position’s hidden vector to a vector of [logits](/ai-glossary/logit/), one score per vocabulary token. Softmax (or log-softmax inside cross-entropy) turns those logits into next-token probabilities.

Some models **tie** this weight matrix to the input [embedding](/ai-glossary/embedding/) table (same parameters, transposed). Others learn a separate projection. Either way, pretraining updates it so the true next token’s logit rises relative to the rest of the vocabulary.
