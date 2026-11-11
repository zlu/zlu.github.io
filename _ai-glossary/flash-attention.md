---
title: Flash Attention
related:
  - Attention
  - Self-Attention
  - Inference
  - Context Window
---
**FlashAttention** is an IO-aware exact [attention](/ai-glossary/attention/) algorithm that tiles computation to reduce HBM reads/writes. It speeds up training and [inference](/ai-glossary/inference/) and lowers activation memory for long sequences, without approximating the attention matrix.

### Why IO matters
Naive attention materializes large $N \times N$ score matrices that thrash GPU memory bandwidth. FlashAttention keeps softmax statistics and outputs in SRAM-friendly tiles, recomputing where needed instead of writing full intermediates to HBM.

### In practice
- Standard in modern LLM stacks when kernels support the GPU/architecture
- Successors (FlashAttention-2/3, vendor fused kernels) further improve occupancy and throughput
- Enables longer [context](/ai-glossary/context-window/) training and serving under the same VRAM budget

Still exact math relative to standard attention (modulo floating-point order); pair with [KV cache](/ai-glossary/kv-cache/) optimizations for decode-side cost.
