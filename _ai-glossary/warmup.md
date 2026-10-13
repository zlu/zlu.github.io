---
title: Warmup
related:
  - Learning Rate
  - Optimizer
---
**Learning-rate warmup** starts training with a smaller LR and ramps up over the first N steps before the main decay schedule.

It stabilizes early updates when gradients or adaptive optimizer statistics are noisy — common practice in transformer pretraining and fine-tuning.
