---
title: Prompt Tuning
related:
  - Prompt Engineering
---
Prompt tuning, in contrast, is a machine-learned, automated approach to crafting prompts. It involves training a small set of parameters (prompt tokens) that are prepended to the input. These tokens are optimized using gradient descent to perform well on a specific downstream task. The base model remains frozen; only the prompt embeddings are updated.

The primary goal of prompt tuning is to adapt a large, pre-trained language model to new tasks without updating the entire model. This method is highly efficient in terms of storage and compute, as it requires updating only a tiny fraction of the parameters.
