---
title: Causal Language Model
description: "Causal language modeling predicts the next token from left context only. Used by GPT, LLaMA, Gemma, and other decoder-only LLMs."
keywords: "causal language modeling, causal language model, causal LM, next-token prediction, GPT, decoder-only"
related:
  - Masked Model
  - Next-Token Prediction
  - Pretraining
---
**Causal language modeling** is training (or running) a model to predict the next token given only the tokens to its left. The model is unidirectional: it never attends to future positions.

Example:

`Input: “The weather is” → Predict: “sunny”`

Decoder-only LLMs are causal language models:

- GPT-2 / GPT-3 / GPT-4-class models
- Gemma
- LLaMA
- Falcon
- Qwen

Contrast with a [masked model](/ai-glossary/masked-model/) (BERT-style), which predicts a blank in the middle of a sequence using both sides.

Related: [next-token prediction](/ai-glossary/next-token-prediction/), [pretraining](/ai-glossary/pretraining/). For adapting an already-trained causal LM to a private corpus, see [continued pretraining](/ai-glossary/continued-pretraining/).
