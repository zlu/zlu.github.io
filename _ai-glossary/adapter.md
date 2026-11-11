---
title: Adapter
related:
  - PEFT
  - LoRA
  - Fine-Tuning
  - Base Model
---
An **adapter** is a small trainable module inserted into a frozen [base model](/ai-glossary/base-model/) so you can specialize without updating (or storing) a full copy of the weights. Classic Houlsby/Pfeiffer adapters add bottleneck MLPs beside Transformer sublayers; [LoRA](/ai-glossary/lora/) is the dominant modern form—low-rank updates on selected projections.

### Why use them
- Far fewer trainable parameters than full [fine-tuning](/ai-glossary/fine-tuning/)
- One shared base + many task/domain adapter packs
- Cheaper multi-tenant serving: swap adapters instead of reloading whole models

### How it shows up
[PEFT](/ai-glossary/peft/) libraries treat LoRA, IA³, and prefix/prompt tuning as adapter-style methods. At serve time you load the base once, attach the right adapter, and optionally merge LoRA into dense weights for latency. Adapters are the practical answer when you need many specialized behaviors on one backbone.
