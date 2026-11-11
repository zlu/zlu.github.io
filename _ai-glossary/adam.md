---
title: Adam
synonyms:
  - AdamW
related:
  - Optimizer
  - Stochastic Gradient Descent
  - Learning Rate
  - Weight Decay
---
**Adam** is an adaptive [optimizer](/ai-glossary/optimizer/): it keeps exponential moving averages of gradients (first moment) and squared gradients (second moment) per parameter, then scales the update like an adaptive [learning rate](/ai-glossary/learning-rate/). It usually needs less LR tuning than plain [SGD](/ai-glossary/stochastic-gradient-descent/).

**AdamW** decouples [weight decay](/ai-glossary/weight-decay/) from the adaptive step (decay applied to weights directly). Prefer AdamW for transformers and LLM fine-tuning; many “Adam” configs in practice are AdamW.

### Knobs that matter
- **Learning rate** — still the main dial; too high → loss spikes
- **β₁ / β₂** — moment decay; defaults (0.9 / 0.95–0.999) often fine
- **Weight decay** — regularization; typical 0.01–0.1 for LLM FT
- **Warmup + schedule** — stabilize early steps at large batch

Adam converges reliably across deep learning but does not remove the need for sensible LR, data quality, and regularization. For full pretraining at huge scale, teams still revisit LR and sometimes SGD-like schedules; for most PEFT/SFT work, AdamW is the default.
