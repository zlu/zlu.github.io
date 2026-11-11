---
title: System Prompt
related:
  - Prompt
  - Prompt Engineering
  - Instruction Tuning
  - Guardrails
  - Tool Calling
  - Chat Template
---
A **system prompt** is a high-priority instruction channel (in chat APIs) that sets role, style, tools, and safety policy for the assistant across turns. It is still part of the overall [prompt](/ai-glossary/prompt/) token sequence, usually placed first in the [chat template](/ai-glossary/chat-template/).

Treat it as product logic: version it, A/B test it, and do not stuff secrets you would not want exposed via prompt extraction. User messages should not silently override critical safety rules; [guardrails](/ai-glossary/guardrails/) outside the model remain necessary.

### Typical contents
- Persona and tone (“concise technical assistant”)
- Tool schemas and when to call them
- Domain constraints and citation rules
- Refusal / safety policy summaries

A long system prompt burns [context window](/ai-glossary/context-window/) budget every turn. Prefer short, stable policy text plus retrieval for volatile facts.
