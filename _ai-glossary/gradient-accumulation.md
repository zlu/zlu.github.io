---
title: Gradient Accumulation
related:
  - Batch Size
  - Mixed Precision
  - Learning Rate
---
**Gradient accumulation** runs several forward/backward passes, summing gradients, then applies one optimizer update. It approximates a larger [batch](/ai-glossary/batch-size/) when the true batch does not fit in VRAM.

### Effective batch size
$$\text{effective batch} \approx \text{micro-batch} \times \text{accumulation steps} \times \text{devices}$$
(for data-parallel training). Keep this number in mind when copying learning-rate recipes—LR often scales with batch size.

### Tradeoffs
- **Pros** — larger effective batches on small GPUs; stable noisy gradients
- **Cons** — more wall-clock per update; BatchNorm-style stats (rare in LLMs) can mismatch true large-batch behavior
- Works well with [mixed precision](/ai-glossary/mixed-precision/) and [gradient checkpointing](/ai-glossary/gradient-checkpointing/) to stretch memory further

Sync/update only after the accumulation window; do not step the optimizer every micro-batch.
