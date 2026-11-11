---
title: Hard Prompt
related:
  - Prompt
  - Soft Prompt
  - Prompt Engineering
  - Prompt Tuning
---
A **hard prompt** is ordinary natural-language text (and special tokens) fed to a model—the human-readable [prompt](/ai-glossary/prompt/) string. Unlike a [soft prompt](/ai-glossary/soft-prompt/), every token maps to the vocabulary; you can copy, version, and audit it as text.

### Why it matters
- Default interface for chat, RAG, and agents
- Portable across models if you adapt the [chat template](/ai-glossary/chat-template/)
- Debuggable: you can read what the model actually saw

### How it shows up
[Prompt engineering](/ai-glossary/prompt-engineering/) edits hard prompts: instructions, few-shot exemplars, tool schemas. [Prompt tuning](/ai-glossary/prompt-tuning/) instead learns soft vectors while keeping the base weights frozen. Hard prompts win on interpretability and zero-shot reuse; soft prompts can beat them on a fixed task when you have labeled data to optimize against.

Use hard prompts when you need reviewable control; reach for soft prompts or [adapters](/ai-glossary/adapter/) when you are optimizing a stable task end-to-end.
