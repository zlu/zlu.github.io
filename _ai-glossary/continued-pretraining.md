---
title: Continued Pretraining
synonyms:
  - CPT
  - Domain-Adaptive Pretraining
related:
  - Pretraining
  - Fine-Tuning
  - Supervised Fine-Tuning
  - Causal Language Model
  - LoRA
---
**Continued pretraining (CPT)** resumes next-token (or masked) language modeling on a new corpus after the original pretraining run — typically to inject a domain, style, or language.

Unlike instruction SFT, CPT trains on raw documents, not chat turns. It is useful when you want the model to *absorb* a library of text; follow with light format SFT so answers stay well-shaped. Watch for catastrophic forgetting of general skills if you train too aggressively without replay or regularization.
