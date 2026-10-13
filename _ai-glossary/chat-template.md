---
title: Chat Template
related:
  - Tokenizer
  - Instruction Tuning
  - System Prompt
---
A **chat template** is the model-specific formatting that wraps roles (system/user/assistant/tool) into the token sequence the instruct model expects.

Using the wrong template is a frequent silent quality killer after fine-tuning or when swapping models. Prefer the tokenizer’s `apply_chat_template`.
