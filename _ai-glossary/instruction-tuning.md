---
title: Instruction Tuning
related:
  - Supervised Fine-Tuning
  - Alignment
  - System Prompt
  - Instruct Model
---
**Instruction tuning** fine-tunes a base LM on diverse (instruction, response) pairs so it follows natural-language commands instead of only continuing raw text. It is the usual path from [base model](/ai-glossary/base-model/) to [instruct/chat](/ai-glossary/instruct-model/) checkpoint.

### Pipeline placement
Often: pretrain → instruction [SFT](/ai-glossary/supervised-fine-tuning/) → optional preference optimization ([DPO](/ai-glossary/dpo/) / [RLHF](/ai-glossary/rlhf/)). Public “Qwen-Instruct”-style releases are typically base models after this stack.

### Domain CPT note
Domain continued pretraining often starts from an instruct checkpoint, then adds light format SFT again after CPT so chat templates and assistant style return. Mix general instruction data with domain tasks to limit forgetting.

Quality tracks data diversity and clarity more than exotic losses—noisy or single-task instruction sets yield narrow, brittle assistants.
