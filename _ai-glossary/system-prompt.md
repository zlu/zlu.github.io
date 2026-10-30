---
title: System Prompt
related:
  - Prompt
  - Prompt Engineering
  - Instruction Tuning
  - Guardrails
---
A **system prompt** is a high-priority instruction channel (in chat APIs) that sets role, style, tools, and safety policy for the assistant across turns. It is still part of the overall [prompt](/ai-glossary/prompt/) token sequence.

Treat it as product logic: version it, test it, and do not stuff secrets you would not want exposed via prompt extraction. User messages should not silently override critical safety rules.
