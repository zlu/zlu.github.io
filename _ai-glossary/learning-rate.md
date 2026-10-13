---
title: Learning Rate
synonyms:
  - LR
related:
  - Optimizer
  - Warmup
  - Gradient Descent
---
The **learning rate** scales how far parameters move per optimizer step. Too high → unstable or divergent training; too low → painfully slow progress.

LLM fine-tuning often uses small LRs (e.g., $1\mathrm{e}{-5}$ to $2\mathrm{e}{-4}$ for LoRA) with warmup and decay schedules. Treat LR as the first knob when loss explodes or refuses to move.
