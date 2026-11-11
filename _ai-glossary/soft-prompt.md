---
title: Soft Prompt
description: "Soft prompt (soft prompting): a learned embedding vector sequence prepended to the input — not readable tokens. Optimized in prompt tuning while the base model stays frozen."
keywords: "soft prompt, soft prompts, soft prompting, prompt tuning, continuous prompt, PEFT"
related:
  - Hard Prompt
  - Prompt Tuning
  - PEFT
  - Embedding
  - Prompt Engineering
---
A **soft prompt** is a learned vector (or short sequence of vectors) in [embedding](/ai-glossary/embedding/) space, prepended to the model input. Unlike a [hard prompt](/ai-glossary/hard-prompt/), it does not map to vocabulary tokens you can read as words.

Soft prompts are the parameters updated in [prompt tuning](/ai-glossary/prompt-tuning/): the base model stays frozen; only the prompt embeddings are trained. They often beat hand-written hard prompts on a target task, at the cost of interpretability—you cannot copy-paste the soft prompt into a chat UI as text.

### Trade-offs
- Tiny trainable footprint vs full fine-tuning or even [LoRA](/ai-glossary/lora/)
- Task-specific; usually not human-editable
- Must be stored and loaded with the serving path that knows how to inject them

Related: [PEFT](/ai-glossary/peft/) methods that adapt a model without full fine-tuning. Contrast with [prompt engineering](/ai-glossary/prompt-engineering/), which edits readable hard prompts only.
