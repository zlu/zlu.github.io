---
title: Domain Adaptation
related:
  - Continued Pretraining
  - Transfer Learning
  - Fine-Tuning
  - LoRA
---
**Domain adaptation** specializes a model to a target distribution—legal, medical, code, or your company’s docs—that differs from pretraining data. The goal is better in-domain accuracy and style without necessarily building a model from scratch ([transfer learning](/ai-glossary/transfer-learning/)).

### Approaches
- **[Continued pretraining](/ai-glossary/continued-pretraining/) (CPT)** on domain text
- **[Fine-tuning](/ai-glossary/fine-tuning/) / SFT** on domain tasks and formats
- **[Adapters](/ai-glossary/adapter/) / [LoRA](/ai-glossary/lora/)** for cheap specialization
- **Retrieval** over a domain corpus ([RAG](/ai-glossary/rag/)) when weights should stay general

### How to judge it
Measure with domain-specific evals, not only general chat vibes. Watch [catastrophic forgetting](/ai-glossary/catastrophic-forgetting/) on out-of-domain probes. Often the winning stack mixes light CPT or LoRA with strong retrieval rather than one giant full fine-tune.
