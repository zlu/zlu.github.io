---
title: Parameter
synonyms:
  - Weights
  - Model Weights
related:
  - Neural Network
  - Fine-Tuning
  - PEFT
  - Quantization
  - Mixture of Experts
---
**Parameters** are the learned weights (and biases) of a model — the numbers updated by the optimizer. LLM size is often quoted as parameter count (7B, 70B), though architecture, data, and training FLOPs matter as much as raw $N$.

### Active vs total
Dense models use (almost) all parameters per token. [MoE](/ai-glossary/mixture-of-experts/) models advertise large total $N$ with smaller *active* parameters per token. [Quantization](/ai-glossary/quantization/) shrinks storage/bandwidth without changing the logical parameter count.

### Training surface
Full [fine-tuning](/ai-glossary/fine-tuning/) updates (nearly) everything. [PEFT](/ai-glossary/peft/) updates a small subset or adds tiny modules (LoRA, adapters), which is usually enough for domain adaptation on modest GPUs.
