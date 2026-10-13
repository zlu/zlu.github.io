---
title: Gradient Accumulation
related:
  - Batch Size
  - Mixed Precision
---
**Gradient accumulation** runs several forward/backward passes, summing gradients, then applies one optimizer update. It approximates a larger batch when the true batch does not fit in VRAM.

Effective batch size ≈ micro-batch × accumulation steps × number of devices (for data parallel).
