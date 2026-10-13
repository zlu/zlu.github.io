---
title: Next-Token Prediction
related:
  - Causal Language Model
  - Pretraining
  - Cross-Entropy
---
**Next-token prediction** trains a model to estimate $P(x_t \mid x_{<t})$. It is the core pretraining (and CPT) objective for decoder LLMs.

Despite its simplicity, it induces rich world knowledge and in-context learning at scale.
