---
title: Feed-Forward Network
synonyms:
  - FFN
  - MLP Block
related:
  - MLP
  - Transformer
  - Activation Function
---
In Transformers, the **feed-forward network** (position-wise [MLP](/ai-glossary/mlp/)) follows attention in each block, expanding to a wider hidden size then projecting back.

Much of an LLM’s parameter count lives in FFNs; MoE variants sparsely activate expert FFNs per token.
