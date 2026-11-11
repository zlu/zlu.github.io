---
title: PEFT
synonyms:
  - Parameter-Efficient Fine-Tuning
related:
  - LoRA
  - Adapter
  - Prompt Tuning
  - Fine-Tuning
  - QLoRA
---
**Parameter-efficient fine-tuning (PEFT)** updates only a small subset of parameters — or adds small modules — while freezing most of the pretrained network.

### Common methods
- **[LoRA](/ai-glossary/lora/) / [QLoRA](/ai-glossary/qlora/)** — low-rank adapters on selected projections
- **[Adapters](/ai-glossary/adapter/)** — small bottleneck modules inserted between layers
- **[Prompt / prefix tuning](/ai-glossary/prompt-tuning/)** — learnable soft prompts; (IA)$^3$ and variants

### Why use it
Lower GPU memory and checkpoint size than full [fine-tuning](/ai-glossary/fine-tuning/); many task adapters can share one base. Quality depends on rank/capacity, target modules, and data — PEFT is not automatically “better,” just cheaper and often good enough.
