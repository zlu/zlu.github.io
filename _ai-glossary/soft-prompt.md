---
title: Soft Prompt
description: "A soft prompt is a learned embedding vector prepended to the input. It is not a readable token string; it is optimized for a task during prompt tuning."
keywords: "soft prompt, soft prompts, soft prompting, prompt tuning, continuous prompt, PEFT"
related:
  - Hard Prompt
  - Prompt Tuning
  - PEFT
---
A **soft prompt** is a learned vector (or short sequence of vectors) in embedding space, prepended to the model input. Unlike a [hard prompt](/ai-glossary/hard-prompt/), it does not map to vocabulary tokens you can read as words.

Soft prompts are the parameters updated in [prompt tuning](/ai-glossary/prompt-tuning/): the base model stays frozen; only the prompt embeddings are trained. They often beat hand-written hard prompts on a target task, at the cost of interpretability.

Related: [PEFT](/ai-glossary/peft/) methods that adapt a model without full fine-tuning.
