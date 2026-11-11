---
title: Residual Connection
synonyms:
  - Skip Connection
related:
  - Transformer
  - Layer Normalization
  - Deep Learning
  - Backpropagation
---
A **residual (skip) connection** adds a block’s input to its output: $y = x + F(x)$. The network learns a residual $F$ instead of a full transformation, which eases [gradient](/ai-glossary/backpropagation/) flow and lets stacks go very deep.

### Why it matters
- Enabled ResNets and every modern [Transformer](/ai-glossary/transformer/) block
- Stabilizes training when $F$ is near identity early on
- Pairs with [layer normalization](/ai-glossary/layer-normalization/) (pre-norm or post-norm)

### How it shows up
In Transformers, attention and MLP sublayers each sit behind a residual: `x = x + Attn(LN(x))`, `x = x + MLP(LN(x))` (pre-norm style). Without skips, deep stacks tend to vanish or explode gradients. When debugging “dead” deep nets, check that residuals and norms are wired as the architecture expects.

Related idea: highway / gated residuals control how much of $F(x)$ mixes in; plain addition remains the LLM default.
