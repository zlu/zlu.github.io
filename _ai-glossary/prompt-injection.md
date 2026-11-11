---
title: Prompt Injection
related:
  - Guardrails
  - System Prompt
  - Agent
  - RAG
  - Tool Calling
---
**Prompt injection** is an attack where untrusted content — a web page, email, uploaded doc, or tool result — contains instructions that try to override the [system prompt](/ai-glossary/system-prompt/), change behavior, or exfiltrate data through the model.

### Why it’s hard
Models are trained to follow natural-language instructions. When retrieved text and user intent share one context window, the model may obey the wrong speaker. [RAG](/ai-glossary/rag/) and [agents](/ai-glossary/agent/) with [tool calling](/ai-glossary/tool-calling/) widen the attack surface.

### Defenses (partial)
Treat retrieved text and tool outputs as hostile data, not trusted commands. Separate instructions from data where possible, minimize tool authority, apply [guardrails](/ai-glossary/guardrails/), and never concatenate untrusted text into a privileged control channel without checks. No single filter is complete — design for least privilege.
