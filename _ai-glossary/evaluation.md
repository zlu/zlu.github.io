---
title: Evaluation
synonyms:
  - Eval
related:
  - Benchmark
  - Perplexity
  - Hallucination
  - Grounding
---
**Evaluation** measures whether a model meets task goals: accuracy, factuality, latency, safety, and user preference. Automated [benchmarks](/ai-glossary/benchmark/) plus human or rubric review beat vibe checks alone.

### What to measure
- **Capability** — task accuracy, [perplexity](/ai-glossary/perplexity/), coding/math suites
- **Reliability** — [hallucination](/ai-glossary/hallucination/) rate, [groundedness](/ai-glossary/grounding/), citation correctness
- **Product** — latency, cost per request, refusal quality, preference win rates

### Domain LLM practice
Fix a prompt suite (in-domain and out-of-domain) and compare base vs adapter with the *same* decoding settings ([temperature](/ai-glossary/temperature/), top-p, max tokens). Hold out eval data from training; regenerate golden answers when the product brief changes.

Offline scores are necessary but incomplete—sample production traces and failure clusters regularly.
