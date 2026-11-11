---
title: Citation
related:
  - Grounding
  - RAG
  - Hallucination
---
In LLM apps, **citations** are pointers from generated claims back to source passages, chunk IDs, or URLs. They help users verify answers and help you debug retrieval mistakes—complementing [grounding](/ai-glossary/grounding/) rather than replacing it.

### Why it matters
- Trust: readers can check claims without re-running the whole stack
- Debug: wrong citation → retrieval/[chunking](/ai-glossary/chunking/) issue; fluent prose with no support → [hallucination](/ai-glossary/hallucination/)
- Fake or invented citations are a failure mode of their own

### How it shows up
Insert explicit chunk IDs into the context and require the model to reference them; post-validate that cited IDs exist and (ideally) entail the claim. UI footnotes without backend validation are theater. Treat citation accuracy as a first-class [evaluation](/ai-glossary/evaluation/) metric for RAG products.
