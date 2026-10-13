---
title: Agent
synonyms:
  - AI Agent
  - LLM Agent
related:
  - Tool Calling
  - RAG
  - LLM Chaining
---
An **LLM agent** uses a language model as a planner/controller that can call tools (search, code, APIs), observe results, and iterate toward a goal — not just answer in one shot.

Agents amplify capability and failure modes: tool misuse, loops, and prompt injection. Keep tools narrowly scoped, add stop conditions, and log every step for debugging.
