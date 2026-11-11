---
title: Beam Search
related:
  - Decoding
  - Sampling
  - Temperature
---
**Beam search** is a [decoding](/ai-glossary/decoding/) algorithm that keeps the top-$B$ partial hypotheses at each step, expanding every beam by possible next tokens and pruning back to $B$ by cumulative score (often length-normalized log-prob). Beam width $B=1$ is greedy decoding.

### Why it matters
- Strong when you want high-probability, structured outputs (MT, ASR, constrained forms)
- Open-ended chat often gets bland or repetitive text compared with nucleus/[temperature](/ai-glossary/temperature/) [sampling](/ai-glossary/sampling/)
- Cost scales with $B$; latency and memory rise accordingly

### When to use it
Prefer beam (or constrained beam) for translation-like tasks and when exactness beats diversity. Prefer sampling for creative assistants. Always judge with task metrics—perplexity of the decoding method alone does not pick the winner.
