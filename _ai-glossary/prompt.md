---
title: Prompt
synonyms:
  - User Prompt
related:
  - System Prompt
  - Hard Prompt
  - Soft Prompt
  - Prompt Engineering
  - Inference
  - Decoding
  - Next-Token Prediction
  - Chat Template
---
A **prompt** is the text (token sequence) you give a language model as input before it generates a continuation. In the simplest case it is a short prefix; the model then predicts the next token, then the next, conditioned on that prefix plus whatever it has already produced ([next-token prediction](/ai-glossary/next-token-prediction/)).

### Chat vs raw
In chat products the prompt is packaged with roles ([system prompt](/ai-glossary/system-prompt/), user, assistant) via a [chat template](/ai-glossary/chat-template/). Underneath it is still tokens into a causal model at [inference](/ai-glossary/inference/) time. [Hard prompts](/ai-glossary/hard-prompt/) are discrete text; [soft prompts](/ai-glossary/soft-prompt/) are learned vectors.

### Craft
[Prompt engineering](/ai-glossary/prompt-engineering/) shapes those inputs; it does not change trained weights. Generation quality also depends on [decoding](/ai-glossary/decoding/) / sampling settings, not only wording.
