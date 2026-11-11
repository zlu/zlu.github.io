---
title: Chat Template
related:
  - Tokenizer
  - Instruction Tuning
  - System Prompt
  - Instruct Model
---
A **chat template** is the model-specific formatting that wraps roles (system / user / assistant / tool) into the exact token sequence an [instruct model](/ai-glossary/instruct-model/) expects—special tokens, headers, and turn delimiters included.

### Why it matters
- Wrong template is a frequent *silent* quality killer after fine-tuning or model swaps
- Training and inference must use the same convention or the model “forgets” how to be an assistant
- [System prompts](/ai-glossary/system-prompt/) only work if the template actually places them where training put them

### How it shows up
Prefer the [tokenizer](/ai-glossary/tokenizer/)’s `apply_chat_template` (or the vendor API’s chat endpoint) over hand-rolled strings. When exporting SFT data, render with the target template before training. Multi-model gateways must map messages per model—not once for all backbones.
