---
title: Supervised Fine-Tuning
synonyms:
  - SFT
related:
  - Fine-Tuning
  - Instruction Tuning
  - Continued Pretraining
  - RLHF
  - Supervised Learning
  - Chat Template
---
**Supervised fine-tuning (SFT)** updates a model on labeled input–output pairs—often instruction/response chats. It teaches *format*, helpfulness style, and task behavior more than it dumps large factual corpora into the weights.

A common recipe for domain specialists: **[CPT](/ai-glossary/continued-pretraining/) on raw text → light SFT on chat** so the model keeps domain knowledge while answering cleanly. Pure SFT on a tiny chat set can teach tone but leave the model factually thin; pure CPT can make the model knowledgeable but chatty in the wrong way.

### What good SFT data looks like
- Clear instructions and high-quality responses
- Coverage of refusal, tool use, and edge cases you care about
- Matching [chat template](/ai-glossary/chat-template/) / special tokens of the base checkpoint

SFT is [supervised learning](/ai-glossary/supervised-learning/) applied to LLMs. Many alignment stacks then continue with preference methods ([RLHF](/ai-glossary/rlhf/), [DPO](/ai-glossary/dpo/)) on top of an SFT checkpoint.
