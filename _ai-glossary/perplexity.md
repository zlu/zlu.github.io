---
title: Perplexity
synonyms:
  - PPL
related:
  - Cross-Entropy
  - Causal Language Model
  - Evaluation
---
**Perplexity** is $\exp(\text{average negative log-likelihood})$ over tokens — how “surprised” a language model is by the text. Lower is better for the training objective.

It is useful for comparing LMs on the same tokenization and domain, but it does not directly measure helpfulness, factuality, or safety. Pair it with task-specific evals.
