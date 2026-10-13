---
title: PEFT
synonyms:
  - Parameter-Efficient Fine-Tuning
related:
  - LoRA
  - Adapter
  - Prompt Tuning
  - Fine-Tuning
---
**Parameter-efficient fine-tuning (PEFT)** updates only a small subset of parameters (or adds small modules) while keeping most of the pretrained network frozen.

Examples: LoRA, adapters, prefix/prompt tuning, (IA)$^3$. PEFT cuts GPU memory and storage versus full fine-tuning and makes it practical to ship many specialized variants of one base model.
