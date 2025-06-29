---
title: Causal Language Model
related:
  - Masked Model
---
A causal language model is a model trained to predict the next word in a sequence, using only the tokens to its left (previous context). It’s unidirectional.

For example:

`Input: “The weather is” →
Predict: “sunny”`

Popular causal LMs:
- GPT-2 / GPT-3
- Gemma
- LLaMA
- Falcon
Contrast this with masked models (like BERT), which predict missing words in the middle.
