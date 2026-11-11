---
title: Guardrails
related:
  - Alignment
  - Prompt Injection
  - System Prompt
  - Tool Calling
---
**Guardrails** are policies and filters around an LLM—input screening, output validation, topic allowlists, PII redaction, and tool-permission checks. They complement (not replace) model [alignment](/ai-glossary/alignment/).

### Layers
- **Input** — jailbreak / [prompt injection](/ai-glossary/prompt-injection/) detectors, length and schema checks
- **Policy** — [system prompts](/ai-glossary/system-prompt/), allow/deny topics, role constraints
- **Output** — toxicity, PII, regex/JSON schema validators, groundedness checks
- **Tools** — least-privilege APIs, confirmation for side effects

### Practice
Defense in depth: validate schemas, constrain [tools](/ai-glossary/tool-calling/), and monitor production traces. Over-strict filters harm usefulness; under-strict ones leak risk. Treat guardrails as product code with tests and metrics, not a one-time prompt tweak.

Fail closed on irreversible actions; fail open (with logging) only where false blocks are costlier than misses.
