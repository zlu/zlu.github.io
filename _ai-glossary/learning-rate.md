---
title: Learning Rate
synonyms:
  - LR
related:
  - Optimizer
  - Warmup
  - Gradient Descent
  - Fine-Tuning
---
The **learning rate** scales how far parameters move per [optimizer](/ai-glossary/optimizer/) step. Too high → unstable or divergent training; too low → painfully slow progress or under-adaptation.

### Typical LLM ranges
Fine-tuning often uses small LRs (e.g., $1\mathrm{e}{-5}$ to $2\mathrm{e}{-4}$ for [LoRA](/ai-glossary/lora/); lower for full-parameter updates) with [warmup](/ai-glossary/warmup/) and decay (cosine, linear). Pretraining uses schedules tuned to token budget and batch size.

### Debugging
Treat LR as the first knob when loss explodes, NaNs, or refuses to move. Pair with grad clipping and sane weight decay. If you change effective [batch size](/ai-glossary/batch-size/) (including via [gradient accumulation](/ai-glossary/gradient-accumulation/)), revisit LR—many recipes assume a specific batch scale.

Log LR alongside loss; a schedule that decays too early can look like “the model stopped learning.”
