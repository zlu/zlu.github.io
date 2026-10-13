---
title: Citation
related:
  - Grounding
  - RAG
  - Hallucination
---
In LLM apps, **citations** are pointers from generated claims back to source passages or URLs. They help users verify answers and debug retrieval mistakes.

Require the model to quote or reference chunk IDs you inserted; post-validate that cited IDs exist. Fake citations are a failure mode of their own.
