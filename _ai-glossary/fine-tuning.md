---
title: Fine-Tuning
related:
  - Supervised Fine-Tuning
  - Continued Pretraining
  - PEFT
  - LoRA
  - Transfer Learning
  - DPO
---
**Fine-tuning** adapts a pretrained model to a downstream task or domain by continuing training on task-specific data. It can mean full-weight updates or [PEFT](/ai-glossary/peft/) methods like [LoRA](/ai-glossary/lora/).

### Common objectives
- **Causal LM on raw text** — [continued pretraining](/ai-glossary/continued-pretraining/) for domain language
- **Supervised chat pairs** — [SFT](/ai-glossary/supervised-fine-tuning/) / [instruction tuning](/ai-glossary/instruction-tuning/)
- **Preference optimization** — [DPO](/ai-glossary/dpo/) / [RLHF](/ai-glossary/rlhf/)
- **Classical heads** — classification or ranking on frozen or lightly tuned backbones

### Choosing a path
Full fine-tuning maximizes capacity but needs more VRAM and risks forgetting. PEFT keeps a shared base and ships small adapters. Always fix decoding and eval suites when comparing base vs tuned—loss alone is not product quality.

Data quality, mix ratios, and [learning rate](/ai-glossary/learning-rate/) usually dominate architecture tweaks.
