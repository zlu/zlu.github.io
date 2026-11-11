---
title: Continued Pretraining
description: "Continued pretraining (CPT) resumes next-token training on a new domain corpus after the original pretraining run. Different from SFT, which uses labeled chat or QA pairs."
keywords: "continued pretraining, CPT, domain-adaptive pretraining, CPT vs SFT, domain LLM"
synonyms:
  - CPT
  - Domain-Adaptive Pretraining
related:
  - Pretraining
  - Fine-Tuning
  - Supervised Fine-Tuning
  - Causal Language Model
  - LoRA
  - Catastrophic Forgetting
  - Domain Adaptation
---
**Continued pretraining (CPT)** resumes next-token (or masked) language modeling on a new corpus after the original [pretraining](/ai-glossary/pretraining/) run, usually to inject a domain, style, or language.

Unlike instruction [SFT](/ai-glossary/supervised-fine-tuning/), CPT trains on raw documents, not chat turns. Use it when you want the model to absorb a library of text; follow with light format SFT so answers stay well formed.

### Risks and mitigations
- [Catastrophic forgetting](/ai-glossary/catastrophic-forgetting/) of general skills if you train too hard without replay or regularization
- Overfitting a tiny domain dump—prefer diverse, deduplicated text
- [LoRA](/ai-glossary/lora/)/[PEFT](/ai-glossary/peft/) CPT can reduce cost and make it easier to keep a general base

Field note (publish when ready): domain-llm Part 1 covers a Qwen3-4B classics CPT attempt and CPT vs SFT.
