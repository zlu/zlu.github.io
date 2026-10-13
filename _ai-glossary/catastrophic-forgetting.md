---
title: Catastrophic Forgetting
related:
  - Continued Pretraining
  - Fine-Tuning
  - Transfer Learning
---
**Catastrophic forgetting** is when fine-tuning on a new distribution erases previously learned skills. Domain CPT can improve in-domain fluency while hurting general benchmarks if overdone.

Mitigations: lower learning rates, LoRA instead of full FT, mixing replay data, early stopping, and evaluating both in-domain and out-of-domain probes.
