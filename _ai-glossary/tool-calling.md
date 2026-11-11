---
title: Tool Calling
synonyms:
  - Function Calling
related:
  - Agent
  - System Prompt
  - Prompt Engineering
  - Guardrails
---
**Tool calling** (function calling) lets a model emit structured requests to external functions—calendars, databases, calculators, retrieval—which the runtime executes and returns as observations.

The model does not execute the tool itself; the host parses a schema-constrained call (often JSON), runs the function, and appends the result to the conversation for the next model turn. That loop is the core of many [agent](/ai-glossary/agent/) designs.

### Reliability checklist
- Clear JSON/schemas and argument types in the [system prompt](/ai-glossary/system-prompt/) or API tool list
- Validation and typed errors back to the model
- Refusal paths when arguments are invalid or tools are unavailable
- Timeouts, auth, and side-effect safeguards outside the model

Tool calling bridges chat LLMs and real systems; treat tool outputs as untrusted input when they re-enter the prompt.
