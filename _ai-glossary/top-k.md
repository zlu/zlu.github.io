---
title: Top-k
related:
  - Top-p
  - Temperature
  - Sampling
---
**Top-$k$ sampling** restricts next-token choices to the $k$ highest-probability tokens, then samples among them (usually after temperature scaling).

Small $k$ reduces unlikely tokens; too-small $k$ can trap the model in dull loops. Often combined with top-$p$ and temperature in LLM APIs.
