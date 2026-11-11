---
title: Batch Size
related:
  - Epoch
  - Gradient Accumulation
  - Learning Rate
  - Mini-Batch
---
**Batch size** is how many examples (or tokens) contribute to one optimizer step. Larger batches yield stabler gradient estimates and better GPU utilization; smaller batches add noise that can regularize but may underuse hardware.

### Why it matters
- Couples tightly with [learning rate](/ai-glossary/learning-rate/): bigger batches often need LR retuning
- Memory bound: activations + KV for long sequences limit what fits
- Reported “global batch” may span devices and accumulation steps

### How it shows up
When VRAM is tight, [gradient accumulation](/ai-glossary/gradient-accumulation/) sums grads over micro-batches before stepping—simulating a larger effective batch. For LLMs, people often count **tokens per step** more carefully than example count. Mismatch between train batch stats and eval can also explain “works in notebook, worse in job” surprises.
