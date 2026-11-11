---
title: Instruct Model
synonyms:
  - Chat Model
related:
  - Base Model
  - Instruction Tuning
  - Chat Template
  - System Prompt
---
An **instruct (chat) model** is a [base LM](/ai-glossary/base-model/) further tuned to follow instructions and multi-turn chat formats. It expects templated roles ([chat template](/ai-glossary/chat-template/)) and usually behaves better as an assistant out of the box.

### How it differs from base
Base models continue raw text; instruct models answer questions, respect [system prompts](/ai-glossary/system-prompt/), and refuse or ask clarifying questions more reliably. Naming varies (`-Instruct`, `-Chat`, `-it`). The upgrade path is typically [instruction tuning](/ai-glossary/instruction-tuning/), often plus preference optimization.

### Domain work
Domain [CPT](/ai-glossary/continued-pretraining/) on instruct checkpoints can scramble formatting and chat habits—hence a small follow-up [SFT](/ai-glossary/supervised-fine-tuning/) after CPT. Always apply the matching tokenizer chat template at train and serve time; mismatched templates look like “the model got dumber.”
