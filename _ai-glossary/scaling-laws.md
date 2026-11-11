---
title: Scaling Laws
related:
  - Pretraining
  - Evaluation
  - Parameter
  - Loss Function
---
**Scaling laws** are empirical relationships between model size ([parameters](/ai-glossary/parameter/)), dataset size, compute, and [loss](/ai-glossary/loss-function/) / downstream performance. They guide how to allocate a [pretraining](/ai-glossary/pretraining/) budget (bigger model vs more data vs more steps).

### What they’re for
Rough forecasts of return on FLOPs; spotting undertrained models (too few tokens for $N$); comparing architectures at matched compute.

### Caveats
Trends bend with data quality, repetition, mixture shifts, and architecture changes (MoE, multimodal). Power-law fits are guides, not guarantees — always validate with held-out [evaluation](/ai-glossary/evaluation/) on your domain, not only predicted loss.
