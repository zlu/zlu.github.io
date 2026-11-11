---
title: Top-k
related:
  - Top-p
  - Temperature
  - Sampling
  - Decoding
  - Softmax
---
**Top-$k$ sampling** restricts next-token choices to the $k$ highest-probability tokens after the [softmax](/ai-glossary/softmax/), then samples among them (usually after [temperature](/ai-glossary/temperature/) scaling).

Small $k$ cuts off the long tail of unlikely tokens and can reduce gibberish; too-small $k$ (or $k=1$, greedy) can trap the model in dull, repetitive loops. Large $k$ approaches full sampling from the tempered distribution.

### How people use it
- Often combined with [top-$p$](/ai-glossary/top-p/) and temperature in LLM APIs
- Creative writing may prefer larger $k$; constrained tasks may prefer smaller $k$
- Evaluate on the task—not only on “sounds nicer”

Top-$k$ is a fixed-size candidate set; top-$p$ adapts the set size to the shape of the distribution. Many stacks apply both: keep tokens that are in the top-$k$ *and* inside the nucleus.
