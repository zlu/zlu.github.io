---
title: Gradient Checkpointing
synonyms:
  - Activation Checkpointing
related:
  - Mixed Precision
  - Backpropagation
---
**Gradient checkpointing** saves memory by not storing all intermediate activations: during backward, selected layers are recomputed. You trade extra compute for the ability to train longer sequences or larger batches.

It is a go-to trick for long-context LoRA/CPT on a single GPU.
