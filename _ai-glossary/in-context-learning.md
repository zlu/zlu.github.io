---
title: In-Context Learning
synonyms:
  - ICL
related:
  - Zero-Shot
  - Few-Shot
  - Prompt Engineering
---
**In-context learning** is the ability of LLMs to adapt to a task from examples or instructions inside the prompt, without updating weights. [Zero-shot](/ai-glossary/zero-shot/), one-shot, and [few-shot](/ai-glossary/few-shot/) prompting are forms of ICL.

### Strengths
- Fast iteration: change demos or instructions without training
- Task switching in one session via [prompt engineering](/ai-glossary/prompt-engineering/)
- Works surprisingly well on format mimicry and simple classification

### Brittleness
Example order, label wording, and delimiters can swing accuracy. Long demos burn [context window](/ai-glossary/context-window/). Hard domain tasks often still need [fine-tuning](/ai-glossary/fine-tuning/) or [RAG](/ai-glossary/rag/).

Treat ICL as a product surface: version prompts, A/B test, and keep a fixed eval suite—do not rely on anecdotal single-prompt wins.
