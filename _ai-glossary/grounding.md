---
title: Grounding
related:
  - RAG
  - Hallucination
  - Citation
  - Tool Calling
---
**Grounding** means tying model outputs to verifiable sources—retrieved documents, tools, or user-provided context—rather than unconstrained parametric recall. Fluent answers can still be ungrounded; measure groundedness explicitly.

### Mechanisms
- **[RAG](/ai-glossary/rag/)** — retrieve passages and require answers to stick to them
- **[Citations](/ai-glossary/citation/)** — point claims at spans or doc IDs
- **[Tool calling](/ai-glossary/tool-calling/)** — fetch live facts (DB, search, calculators) before answering

### Evaluation
Score whether claims are supported by supplied evidence, not only whether they sound right. Track unsupported claims ([hallucination](/ai-glossary/hallucination/)) separately from retrieval misses. System prompts that say “use only the context” help but are not sufficient without checks.

Grounding improves trust for enterprise Q&A; it does not replace [alignment](/ai-glossary/alignment/) or [guardrails](/ai-glossary/guardrails/) for safety policy.
