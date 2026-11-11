---
title: Prompt Tuning
description: "Prompt tuning adapts a frozen LLM by training a small soft-prompt embedding prepended to inputs. Only those prompt parameters are updated."
keywords: "prompt tuning, soft prompt, PEFT, parameter-efficient fine-tuning, frozen LLM"
related:
  - Soft Prompt
  - Prompt Engineering
  - PEFT
  - Hard Prompt
  - LoRA
  - Fine-Tuning
---
**Prompt tuning** adapts a large pretrained language model to a downstream task without updating the full weights. You train a small set of [soft prompt](/ai-glossary/soft-prompt/) embeddings prepended to each input. The base model stays frozen; gradients update only the prompt parameters.

### Goals
- Task adaptation with a tiny trainable footprint
- One shared frozen backbone, many task-specific soft prompts
- Less storage and compute than full [fine-tuning](/ai-glossary/fine-tuning/)

### Contrast
- [Prompt engineering](/ai-glossary/prompt-engineering/) — hand-written [hard prompts](/ai-glossary/hard-prompt/); no training
- [LoRA](/ai-glossary/lora/) / other [PEFT](/ai-glossary/peft/) — updates low-rank (or adapter) weights inside layers, usually stronger on hard tasks
- Full fine-tuning — all parameters move; highest capacity and cost

Prompt tuning shines when you need many cheap task heads on one frozen server model. Soft prompts are not human-readable; treat them like small binary adapters in your deploy path.
