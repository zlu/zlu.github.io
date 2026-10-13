---
title: Adapter
related:
  - PEFT
  - LoRA
  - Fine-Tuning
---
**Adapters** are small trainable modules inserted into a frozen network (classic Houlsby/Pfeiffer adapters, or LoRA-style updates). Each task can have its own adapter pack.

They enable modular specialization and cheaper multi-tenant serving compared with full fine-tuned copies of the base model.
