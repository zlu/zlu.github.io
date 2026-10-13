---
title: Batch Size
related:
  - Epoch
  - Gradient Accumulation
  - Learning Rate
---
**Batch size** is the number of examples (or tokens) per optimizer step. Larger batches give stabler gradients and higher GPU utilization; smaller batches can regularize but may underuse hardware.

When memory is tight, **gradient accumulation** simulates larger batches by summing gradients over micro-batches before stepping.
