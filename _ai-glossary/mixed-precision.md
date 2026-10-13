---
title: Mixed Precision
synonyms:
  - AMP
  - FP16 Training
  - BF16
related:
  - Gradient Checkpointing
  - Quantization
---
**Mixed-precision training** computes many ops in FP16 or BF16 while keeping a master copy of weights in FP32 (or using BF16 throughout on supporting hardware). It cuts memory and often speeds training on modern GPUs.

BF16 is numerically friendlier than FP16 for many LLM runs. Pair with loss scaling when using FP16.
