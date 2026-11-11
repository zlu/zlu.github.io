---
title: Few-Shot
synonyms:
  - Few-Shot Learning
  - N-Shot
related:
  - Zero-Shot
  - In-Context Learning
  - Prompt Engineering
---
**Few-shot** prompting includes a handful of input–output demonstrations in the prompt so the model infers the pattern. It sits between [zero-shot](/ai-glossary/zero-shot/) (no examples) and [fine-tuning](/ai-glossary/fine-tuning/) (weight updates), and is a form of [in-context learning](/ai-glossary/in-context-learning/).

### When it helps
Use few-shot when the task format is unusual, labels are idiosyncratic, or zero-shot instructions are ambiguous. Consistent schema, delimiters, and representative edge cases matter more than dumping many near-duplicates.

### Limits
- Examples consume [context window](/ai-glossary/context-window/) budget
- Order and wording can swing accuracy (brittleness)
- Hard tasks or strict domain jargon may still need SFT or [RAG](/ai-glossary/rag/)

Prefer a small, curated set over long noisy demos; measure gains on a fixed eval before shipping a prompt change.
