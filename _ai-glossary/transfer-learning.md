---
title: Transfer Learning
related:
  - Fine-Tuning
  - Pretraining
  - Domain Adaptation
  - Continued Pretraining
  - PEFT
---
**Transfer learning** reuses knowledge from a model trained on one task or domain to improve learning on another—typically via pretrained weights plus [fine-tuning](/ai-glossary/fine-tuning/).

Modern NLP is almost entirely transfer learning: [pretrain](/ai-glossary/pretraining/) once on broad data, adapt cheaply to many tasks. That adaptation may be full fine-tuning, [PEFT](/ai-glossary/peft/) ([LoRA](/ai-glossary/lora/)), [continued pretraining](/ai-glossary/continued-pretraining/) on domain text, or even prompt-only methods.

### Why it works
- Early layers learn reusable features (syntax, morphology, common sense patterns)
- Downstream data is usually too small to train a large model from scratch
- Shared checkpoints amortize compute across products

Watch for negative transfer (source domain hurts the target) and [catastrophic forgetting](/ai-glossary/catastrophic-forgetting/) when you adapt aggressively. Validate on the real target distribution, not only on the pretraining domain.
