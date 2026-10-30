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
---
A **prompt** is the text (token sequence) you give a language model as input before it generates a continuation. In the simplest case it is a short prefix such as `Rain falls`. The model then predicts the next token, then the next, conditioned on that prefix plus whatever it has already produced.

In chat products the prompt is usually packaged with roles ([system prompt](/ai-glossary/system-prompt/), user, assistant) via a chat template. Underneath, it is still tokens into a causal model. [Prompt engineering](/ai-glossary/prompt-engineering/) is the craft of writing those inputs; it does not change the trained weights.
