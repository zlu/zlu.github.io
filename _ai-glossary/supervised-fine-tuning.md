---
title: Supervised Fine-Tuning
synonyms:
  - SFT
related:
  - Fine-Tuning
  - Instruction Tuning
  - Continued Pretraining
  - RLHF
---
**Supervised fine-tuning (SFT)** updates a model on labeled input–output pairs — often instruction/response chats. It teaches *format*, helpfulness style, and task behavior more than it dumps large factual corpora into the weights.

A common recipe for domain specialists: **CPT on raw text → light SFT on chat** so the model keeps domain knowledge while answering cleanly.
