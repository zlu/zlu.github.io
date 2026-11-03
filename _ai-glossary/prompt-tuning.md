---
title: Prompt Tuning
description: "Prompt tuning adapts a frozen LLM by training a small soft-prompt embedding prepended to inputs. Only those prompt parameters are updated."
keywords: "prompt tuning, soft prompt, PEFT, parameter-efficient fine-tuning, frozen LLM"
related:
  - Soft Prompt
  - Prompt Engineering
  - PEFT
  - Hard Prompt
---
**Prompt tuning** adapts a large pretrained language model to a downstream task without updating the full weights. You train a small set of [soft prompt](/ai-glossary/soft-prompt/) embeddings prepended to each input. The base model stays frozen; gradients update only the prompt parameters.

Goals:

- Task adaptation with a tiny trainable footprint
- One shared frozen backbone, many task-specific soft prompts
- Less storage and compute than full fine-tuning

Contrast with [prompt engineering](/ai-glossary/prompt-engineering/) (hand-written [hard prompts](/ai-glossary/hard-prompt/)) and with weight-updating methods such as [LoRA](/ai-glossary/lora/) under [PEFT](/ai-glossary/peft/).
