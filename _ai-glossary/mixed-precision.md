---
title: Mixed Precision
synonyms:
  - AMP
  - FP16 Training
  - BF16
related:
  - Gradient Checkpointing
  - Quantization
  - Tensor
---
**Mixed-precision training** runs most matmuls and activations in FP16 or BF16 while keeping a higher-precision master copy of weights (typically FP32) for stable updates. On supporting hardware, BF16 can be used throughout without loss scaling.

### Why use it
- Roughly halves activation / weight memory vs FP32
- Often faster on Tensor Cores / matrix units
- Enables larger models or batch sizes on the same GPU

### FP16 vs BF16
FP16 needs **loss scaling** (dynamic or static) to avoid underflow in gradients. BF16 keeps FP32-like exponent range and is usually the default for modern LLM training. Do not confuse training mixed precision with post-training [quantization](/ai-glossary/quantization/) (INT8/INT4 for inference). Pair with [gradient checkpointing](/ai-glossary/gradient-checkpointing/) when memory is still tight.
