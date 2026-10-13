---
title: LoRA
synonyms:
  - Low-Rank Adaptation
related:
  - PEFT
  - QLoRA
  - Fine-Tuning
  - Adapter
---
**LoRA** (Low-Rank Adaptation) freezes a pretrained model and injects small trainable low-rank matrices into selected layers (often attention and MLP projections). Instead of updating full weight $W$, you learn $\Delta W = BA$ where $B$ and $A$ have small rank $r$.

### Why use it
- Far fewer trainable parameters than full fine-tuning
- Multiple task adapters can share one base model
- Easy to merge or swap adapters at serve time

### Knobs that matter
- **Rank ($r$)** — capacity of the adapter
- **Target modules** — which projections get LoRA
- **Alpha / dropout** — scaling and regularization

LoRA is the workhorse PEFT method for continued pretraining and domain SFT on small GPUs.
