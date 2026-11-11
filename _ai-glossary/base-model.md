---
title: Base Model
related:
  - Instruct Model
  - Pretraining
  - Fine-Tuning
  - Continued Pretraining
---
A **base model** is a pretrained language model *before* instruction/chat alignment—strong at continuation and completion, weaker at following assistant conventions out of the box. Weights come from [pretraining](/ai-glossary/pretraining/) (next-token or similar) on large corpora.

### Why it matters
- Starting checkpoint choice changes CPT/SFT recipes
- Base models are often better substrates for domain [continued pretraining](/ai-glossary/continued-pretraining/) than chat-tuned ones
- Serving a base model as a “chatbot” without a template usually feels broken

### How it shows up
Release names often pair `*-base` with `*-instruct` / `*-chat`. Some domain recipes CPT a base then lightly SFT; others start from an [instruct model](/ai-glossary/instruct-model/) and SFT again. Always confirm which checkpoint you have, which [chat template](/ai-glossary/chat-template/) applies, and whether license/terms allow your use.
