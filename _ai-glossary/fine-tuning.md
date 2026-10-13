---
title: Fine-Tuning
related:
  - Supervised Fine-Tuning
  - Continued Pretraining
  - PEFT
  - LoRA
  - Transfer Learning
---
**Fine-tuning** adapts a pretrained model to a downstream task or domain by continuing training on task-specific data. It can mean full-weight updates or PEFT methods like LoRA.

Choose the objective carefully: causal LM on raw text (CPT), supervised chat pairs (SFT), preference optimization (DPO/RLHF), or classical classification heads — each encodes different skills.
