---
title: Prompt Injection
related:
  - Guardrails
  - System Prompt
  - Agent
---
**Prompt injection** is an attack where untrusted content (a web page, email, or uploaded doc) contains instructions that try to override the system prompt or exfiltrate data via the model.

Treat retrieved text and tool outputs as hostile. Separate instructions from data, minimize tool authority, and never concatenate untrusted text into a privileged control channel without sanitization.
