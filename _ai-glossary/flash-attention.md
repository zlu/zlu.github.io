---
title: Flash Attention
related:
  - Attention
  - Self-Attention
  - Inference
---
**FlashAttention** is an IO-aware exact attention algorithm that tiles computation to reduce HBM reads/writes, speeding up training and inference and lowering memory for long sequences.

It (and successors) are standard in modern LLM stacks whenever hardware supports the kernels.
